import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dilation_rate, n_filters, kernel_size, dropout_rate=0.15):
        super().__init__()
        in_channels = 20
        self.res = nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size=1, padding='same')
        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size=kernel_size, dilation=dilation_rate, padding='same')
        self.conv_2 = nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size=kernel_size, dilation=dilation_rate*2, padding='same')
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.15)
        self.conv_3 = nn.Conv1d(in_channels=2*n_filters, out_channels=n_filters, kernel_size=1, padding='same')

        return

    def forward(self, x):
        res_x = self.res(x)
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(x)
        concat = torch.cat([conv_1, conv_2], dim=1)
        out = self.elu(concat)
        out = self.dropout(out)
        out = self.conv_3(out)

        return res_x + out, out

class TCN(nn.Module):
    def __init__(self, n_filters, kernel_size, n_dilations, dropout_rate):
        super().__init__()
        dilations = [2**i for i in range(n_dilations)]

        self.tcn_layers = nn.ModuleDict({})

        for idx, d in enumerate(dilations):
            self.tcn_layers[f"tcn_{idx}"] = ResBlock(d, n_filters, kernel_size, dropout_rate)

        self.activation = nn.ELU()

        return

    def forward(self, x):
        skip_connections = []
        for tcn_i in self.tcn_layers:
            x, skip_out = self.tcn_layers[tcn_i](x)
            skip_connections.append(skip_out)

        x = self.activation(x)

        skip = torch.stack(skip_connections, dim=-1).sum(dim=-1)

        return x, skip


class MultiTracker(nn.Module):
    def __init__(self, n_filters=20, n_dilations=11, kernel_size=5, dropout_rate=0.15):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(3, 3), stride=1, padding="valid")
        self.elu_1 = nn.ELU()
        self.mp_1 = nn.MaxPool2d((1, 3))
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.conv_2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=(1, 10), padding="valid")
        self.elu_2 = nn.ELU()
        self.mp_2 = nn.MaxPool2d((1, 3))
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.conv_3 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=(3, 3), padding="valid")
        self.elu_3 = nn.ELU()
        self.mp_3 = nn.MaxPool2d((1, 3))
        self.dropout_3 = nn.Dropout(dropout_rate)

        self.tcn = TCN(n_filters, kernel_size, n_dilations, dropout_rate)

        self.beats_dropout = nn.Dropout(dropout_rate)
        self.beats_dense = nn.Linear(n_filters, 1)
        self.beats_act = nn.Sigmoid()

        self.downbeats_dropout = nn.Dropout(dropout_rate)
        self.downbeats_dense = nn.Linear(n_filters, 1)
        self.downbeats_act = nn.Sigmoid()

        self.tempo_dropout = nn.Dropout(dropout_rate)
        self.tempo_dense = nn.Linear(n_filters, 300)
        self.tempo_act = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.elu_1(x)
        x = self.mp_1(x)
        x = self.dropout_1(x)

        x = self.conv_2(x)
        x = self.elu_2(x)
        x = self.mp_2(x)
        x = self.dropout_2(x)

        x = self.conv_3(x)
        x = self.elu_3(x)
        x = self.mp_3(x)
        x = self.dropout_3(x)

        x = torch.squeeze(x, -1)

        x, skip = self.tcn(x)

        x = x.transpose(-2,-1)

        beats = self.beats_dropout(x)
        beats = self.beats_dense(beats)
        beats = self.beats_act(beats)

        downbeats = self.downbeats_dropout(x)
        downbeats = self.downbeats_dense(downbeats)
        downbeats = self.downbeats_act(downbeats)

        activations = {}
        activations["beats"] = beats
        activations["downbeats"] = downbeats

        return activations

    def load_weights(self, model_path, device):
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[len("model."):]
            new_state_dict[k] = v

        self.load_state_dict(new_state_dict)
        return
