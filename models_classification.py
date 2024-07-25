import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    
    def __init__(
        self,
        input_size  : int = 1,
        output_size : int = 1,
        hidden_size : int = 2,
        num_layers  : int = 1,
        dropout     : float = 0.1,
        device      : torch.device = torch.device('cpu')
    ):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        # Fully connected layer
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
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



# LSTM + 1D smoothing convolutional layers
class Astice(nn.Module):
    
    def __init__(
        self,
        input_size  : int = 1,
        output_size : int = 1,
        hidden_size : int = 2,
        lstm_size : int = 2,
        lstm_layers : int = 1,
        smooth_conv_channels : int = 1,
        drouput_rate : float = 0.2
    ):
        super(Astice, self).__init__()
        # Save model hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.smooth_conv_channels = smooth_conv_channels
        self.drouput_rate = drouput_rate
        
        # Smooth convolutional layers run in parallel
        self.smooth_k_1 = 3
        self.smooth_k_2 = 5
        self.smooth_k_3 = 7
        self.conv1 = nn.Conv1d(input_size, smooth_conv_channels, kernel_size=self.smooth_k_1, padding=self.smooth_k_1//2)
        self.conv2 = nn.Conv1d(input_size, smooth_conv_channels, kernel_size=self.smooth_k_2, padding=self.smooth_k_2//2)
        self.conv3 = nn.Conv1d(input_size, smooth_conv_channels, kernel_size=self.smooth_k_3, padding=self.smooth_k_3//2)
        self.fcnn_1 = nn.Sequential(
            nn.Linear(input_size+3*smooth_conv_channels, lstm_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, lstm_size)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(lstm_size, lstm_size, lstm_layers, batch_first=True, bidirectional=False)
        
        
        # Output layer
        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.drouput_rate)
        self.fcnn_input_size = lstm_size * lstm_layers
        #self.fcnn_2 = nn.Sequential(
        #    nn.Linear(self.fcnn_input_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, output_size)
        #)
        self.n_features = 5
        self.fcnn_out = nn.ModuleList([
            nn.Sequential(
                self.dropout,
                nn.Linear(self.fcnn_input_size, hidden_size),
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
      
      
      
     
# LOB in progress...
class Aragosta(nn.Module):

    def __init__(
        self,
        input_size  : int = 1,
        output_size : int = 1,
        hidden_size : int = 2,
        lstm_size : int = 2,
        lstm_layers : int = 1,
        conv_channels : int = 1,
        inception_channels : int = 1,
        dropout_rate : float = 0.1
    ):
        super(Aragosta, self).__init__()
        # Save model hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.conv_channels = conv_channels
        self.inception_channels = inception_channels
        self.dropout_rate = dropout_rate
        
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # 2D convolutional layers for feature extraction
        self.cnn_fe_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.conv_channels, kernel_size=(1, 2), stride=(1,2)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.conv_channels),
        )
        self.cnn_fe_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=(1, 2), stride=(1,2)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.conv_channels),
        )
        self.cnn_fe_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=(1, 10), stride=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.conv_channels),
        )
        
        # Inception module
        self.inception_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_channels, out_channels=self.inception_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.inception_channels),
            nn.Conv2d(in_channels=self.inception_channels, out_channels=self.inception_channels, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.inception_channels),
        )
        self.inception_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_channels, out_channels=self.inception_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.inception_channels),
            nn.Conv2d(in_channels=self.inception_channels, out_channels=self.inception_channels, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.inception_channels),
        )
        self.inception_3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=self.conv_channels, out_channels=self.inception_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.inception_channels),
        )
        self.inception_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_channels, out_channels=self.inception_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.inception_channels),
        )
        self.fcnn_1 = nn.Sequential(
            nn.Linear(self.inception_channels*4, self.inception_channels*2),
            nn.LeakyReLU(),
            self.dropout,
            nn.Linear(self.inception_channels*2, lstm_size)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(lstm_size, lstm_size, lstm_layers, batch_first=True, bidirectional=False)
        
        # Output layer
        self.fcnn_input_size = lstm_size * lstm_layers
        self.fcnn_2 = nn.Sequential(
            self.dropout,
            nn.Linear(self.fcnn_input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.softmax = nn.Softmax(dim=1)
        
        self.init_weights()
        
    def forward(self, x):
        batch_size = x.size(0)
    
        # 2D convolutional layers for feature extraction
        x = self.cnn_fe_1(x)
        x = self.cnn_fe_2(x)
        x = self.cnn_fe_3(x)
        
        # Inception module
        x_inception_1 = self.inception_1(x)
        x_inception_2 = self.inception_2(x)
        x_inception_3 = self.inception_3(x)
        x_inception_4 = self.inception_4(x)
        x = torch.cat((x_inception_1, x_inception_2, x_inception_3, x_inception_4), dim=1)
        x = x.permute(0, 2, 1, 3).squeeze(3)
        x = self.fcnn_1(x)
        
        # Forward propagate LSTM
        _, (h_n, _) = self.lstm(x)
        
        # Decode the hidden state of the last time step
        x = h_n.permute(1, 0, 2).reshape(batch_size, -1)
        
        # Decode the hidden state of the last time step
        out = self.fcnn_2(x).unsqueeze(1).unsqueeze(1)
        
        out = out.squeeze()
        out = self.softmax(out)
        
        return out
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)












