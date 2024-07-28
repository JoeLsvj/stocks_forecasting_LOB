import torch
import torch.nn as nn
import torch.nn.functional as F


# 1 - Basic LSTM - ( hidden_size, dropout_rate )
# hidden_size       32:128
# dropout_rate     0.0:0.4
class LSTM(nn.Module):
    
    def __init__(
        self,
        input_size  : int = 1,
        output_size : int = 1,
        hidden_size : int = 64,
        num_layers  : int = 2,
        dropout_rate: float = 0.15,
    ):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        # Fully connected layer
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fcnn_input_size = self.hidden_size*self.num_layers
        self.fcnn_1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            self.activation,
            self.dropout,
            nn.Linear(hidden_size, hidden_size),
        )

        self.softmax = nn.Softmax(dim=1)
        
        # Output
        self.n_features = 5
        self.fcnn_out = nn.ModuleList([
            nn.Sequential(
                self.dropout,
                nn.Linear(self.fcnn_input_size, hidden_size),
                self.activation,
                nn.Linear(hidden_size, 3)
            ) for _ in range(self.n_features)
        ])
        
        # Initialize weights
        self.init_wheights()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Fully connected layer 1
        x = self.fcnn_1(x)
        
        # Forward propagate LSTM
        _, (hn, _) = self.lstm(x)
        
        # Decode the hidden state of the last time step
        # seq_len, batch, size -> batch, seq_len, size -> batch, size*seq_len
        x = hn.permute(1, 0, 2).reshape(batch_size, -1)
        
        # Apply the output layers
        out = [ fcnn(x) for fcnn in self.fcnn_out ]
        out = [ self.softmax(o) for o in out ]
        out = torch.cat(out, dim=1)
        
        out = out.view(batch_size, self.n_features, 3)
        
        return out

    def init_wheights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)



# 2 - Astice - ( hidden_size, lstm_size, smooth_conv_channels, dropout_rate )
# hidden_size           32:128
# lstm_size             32:128
# smooth_conv_channels   4:64
# dropout_rate         0.0:0.4
class Astice(nn.Module):
    
    def __init__(
        self,
        input_size  : int = 1,
        output_size : int = 1,
        hidden_size : int = 64,
        lstm_size : int = 64,
        smooth_conv_channels : int = 8,
        dropout_rate : float = 0.15
    ):
        super(Astice, self).__init__()
        # Save model hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        self.lstm_layers = 2
        self.smooth_conv_channels = smooth_conv_channels
        self.dropout_rate = dropout_rate
        
        # Smooth convolutional layers run in parallel
        self.smooth_k_1 = 3
        self.smooth_k_2 = 5
        self.smooth_k_3 = 7
        self.conv1 = nn.Conv1d(input_size, smooth_conv_channels, kernel_size=self.smooth_k_1, padding=self.smooth_k_1//2)
        self.conv2 = nn.Conv1d(input_size, smooth_conv_channels, kernel_size=self.smooth_k_2, padding=self.smooth_k_2//2)
        self.conv3 = nn.Conv1d(input_size, smooth_conv_channels, kernel_size=self.smooth_k_3, padding=self.smooth_k_3//2)
        self.fcnn_1 = nn.Sequential(
            nn.Linear(input_size+3*smooth_conv_channels, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, lstm_size)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(lstm_size, lstm_size, self.lstm_layers, batch_first=True, bidirectional=False)
        
        
        # Output layer
        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.n_features = 5
        self.fcnn_out = nn.ModuleList([
            nn.Sequential(
                self.dropout,
                nn.Linear(lstm_size * self.lstm_layers, hidden_size),
                self.activation,
                nn.Linear(hidden_size, 3)
            ) for _ in range(self.n_features)
        ])

        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Smooth convolutional layers
        x = x.permute(0, 2, 1)
        x_smooth_1 = self.conv1(x)
        x_smooth_2 = self.conv2(x)
        x_smooth_3 = self.conv3(x)
    
        # Concatenate the outputs of the smooth convolutional layers
        x = torch.cat((x, x_smooth_1, x_smooth_2, x_smooth_3), dim=1)
        x = x.permute(0, 2, 1)
        x = self.fcnn_1(x)
        
        # Forward propagate LSTM
        _, (h_n, _) = self.lstm(x)
        
        
        # Decode the hidden state of the last time step
        x = h_n.permute(1, 0, 2).reshape(batch_size, -1)
        
        # Decode the hidden state of the last time step
        out = [ fcnn(x) for fcnn in self.fcnn_out ]
        out = [ self.softmax(o) for o in out ]
        out = torch.cat(out, dim=1)
        
        out = out.view(batch_size, self.n_features, 3)
        
        return out
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
      
      
      
# 3 - Marmotta - ( hidden_size, dropout_rate )
class Marmotta(nn.Module):

    def __init__(
            self,
            input_size = 1,
            hidden_size = 64,
            output_size = 1,
            seq_len_in = 1,
            seq_len_out = 5,
            # LSTM layers
            lstm_layers = 2,
            # FCNN
            dropout_rate = 0.15,
            device = torch.device('cpu')
    ):
        super(Marmotta, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout_rate
        self.device = device        
        
        # Encoder
        self.fcnn_1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )
        self.lstm_enc = nn.LSTM(hidden_size, hidden_size, lstm_layers, batch_first=True)
        
        # Decoder
        self.lstm_dec = nn.LSTM(hidden_size, hidden_size, lstm_layers, batch_first=True)
        self.fcnn_2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )
        self.fcnn_3 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size*self.lstm_layers, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.fcnn_4 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.seq_len_in*self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.fcnn_5 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(3*self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )
        self.fcnn_list = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, output_size)
            ) for _ in range(seq_len_out)
        ])
    
    def encoder(self, x):
        # FCNN
        x = self.fcnn_1(x)
        # LSTM
        outputs, (hn, cn) = self.lstm_enc(x)

        return outputs, (hn, cn)
    
    def decoder(self, x, states):
        # LSTM
        outputs, (hn, _) = self.lstm_enc(x, states)
        # FCNN
        outputs = self.fcnn_2(outputs)
        
        return outputs, hn        
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encoder
        z, states = self.encoder(x)
        # Decoder
        h, hn = self.decoder(z, states)
        # hidden representation
        h = h.reshape(batch_size, -1)
        _h = self.fcnn_4(h)
        
        
        # Context vector
        hn = hn.permute(1, 0, 2).reshape(batch_size, -1)
        c = self.fcnn_3(hn)
        
        # Predict every time step
        y = []
        prev = torch.zeros(batch_size, self.hidden_size).to(self.device)
        for i in range(self.seq_len_out):
            
            v = torch.cat([_h, c, prev], dim=1)
            v = self.fcnn_5(v)
            prev = v
            y_hat = self.fcnn_list[i](v).unsqueeze(1)
            y.append(y_hat)

        # Concatenate the predictions
        y = torch.stack(y, dim=1).squeeze(2)
        
        return y


