import torch
import torch.nn as nn
import torch.nn.functional as F

# simple LSTM model
class LSTM1(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'LSTM1'
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        # input x has shape [batch, timestamp, n features = input size]
        batchsize = x.shape[0]
        # linear layer 1
        x = self.linear_1(x)
        x = self.relu(x)
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Output shape of the lstm_out :  torch.Size([batch, timestamp window, hidden size]) if batch_first = True
        # Output shape of the h_n :  torch.Size([num layers, batch, hidden size])
        # Output shape of the c_n :  torch.Size([num_layers, batch, hidden size])
        # the last element (in timesteps axis) of lstm_out is h_n, i.e lstm_out[:, -1, :] == h_n
        # permute() changes the position of the dimensions. On a tensor of size [num layers, batch, hidden size] 
        # (as h_n), permute(1,0,2) transforms in [batch, num layers, hidden size]
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        # linear layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        #predictions = torch.squeeze(predictions)
        return predictions.unsqueeze(1)


# double LSTM model
class LSTM2(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'LSTM2'
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.linear_2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear_3 = nn.Linear(num_layers*hidden_layer_size, output_size)
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        # input x has shape [batch, lookback_period, n features = input size]
        batchsize = x.shape[0]
        # linear layer 1
        x = self.linear_1(x)
        x = self.relu(x)
        # lstm_out = [batch, lookback_period, encoder_hidden_size]
        # h_n = [num_layers, batch, encoder_hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # linear layer 2
        x = self.linear_2(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.dropout(x)
        predictions = self.linear_3(x)
        #predictions = torch.squeeze(predictions)
        return predictions


# encoder-decoder LSTM with attention mechanism
class LSTMWithAttention(nn.Module):

    def __init__(self, input_size=1, 
                encoder_hidden_size=32, 
                encoder_layers=2, 
                decoder_hidden_size=32, 
                decoder_layers=2,
                output_size=1,
                dropout=0.2
            ):
        super(LSTMWithAttention, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'LSTMAttention'
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_laye = decoder_layers
        self.output_size = output_size
        self.dropout = dropout

        self.encoder     = nn.LSTM(input_size, encoder_hidden_size, encoder_layers, batch_first=True)
        self.decoder     = nn.LSTM(decoder_hidden_size, decoder_hidden_size, decoder_layers, batch_first=True)
        self.relu        = nn.ReLU()
        self.dropout     = nn.Dropout(dropout)
        self.linear_out  = nn.Linear(decoder_layers*decoder_hidden_size, output_size)

        self.init_weights()

    def Attention(self, encoder_outputs, decoder_hidden=None, decoder_hidden_size=None):
        # encoder_outputs: (batch, lookback_period, hidden_dim)
        # decoder_hidden:  (batch, hidden_dim)
        # 1st way to compute context vector:
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)             # (batch, lookback_period)
        attention_weights = F.softmax(scores, dim=1)                                            # (batch, lookback_period)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden_dim)
        # 2nd way to compute context vector:
        #linear_attention = nn.Linear(decoder_hidden_size, 1)
        #attention_weights = torch.softmax(linear_attention(encoder_outputs).squeeze(-1), dim=-1)
        #context_vector = torch.sum(encoder_outputs * attention_weights.unsqueeze(-1), dim=1)
        return context_vector, attention_weights

    def forward(self, x):
        # x = [batch, lookback_period, num_features]
        batchsize = x.shape[0]
        # shapes of the following outputs are:
        # encoder_outputs = [batch, lookback_period, encoder_hidden_size]
        # encoder_h_c = [num_layers, batch, encoder_hidden_size]
        encoder_outputs, (encoder_h_c, encoder_c_n) = self.encoder(x)
        # context_vector = [batch, encoder_hidden_size]
        context_vector, attn_weights = self.Attention(encoder_outputs, encoder_h_c[-1], self.encoder_hidden_size)

        # version 1
        # dec_input -> [num_layers+1, batch, encoder_hidden_size] +1 for the context vector
        #dec_input = torch.cat([encoder_h_c, context_vector.unsqueeze(0)], dim=0)
        ## dec_input -> [batch, num_layers+1, encoder_hidden_size]
        #dec_input = dec_input.permute(1, 0, 2)

        # version 2
        # dec_input -> [batch, lookback_period+1, encoder_hidden_size]
        dec_input = torch.cat([encoder_outputs, context_vector.unsqueeze(1)], dim=1)

        # lstm decoder layer
        output, (decoder_h_c, decoder_c_n) = self.decoder(dec_input) # encoder_h_c, encoder_c_n
        x = decoder_h_c.permute(1, 0, 2).reshape(batchsize, -1)
        # final linear layer
        x = self.dropout(x)
        predictions = self.linear_out(x)
        #predictions = torch.squeeze(predictions)
        return predictions

    def init_weights(self):
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
        for name, param in self.decoder.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

class LobCNN(nn.Module):
    def __init__(   self,
                    lookback_period = 100,
                    inner_convolutional_layers = 2,
                    smoothing_percentage = 0.06,
                    convolution_channels = 32,
                    inception_channels = 64,
                    lstm_layers = 2,
                    output_size = 1,
                    regression = True,
                ):
        super(LobCNN, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'LobCNN'

        self.regression = regression
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        self.convolution_channels = convolution_channels
        self.inception_channels = inception_channels
        self.smoothing_kernel_size = int(smoothing_percentage * lookback_period/inner_convolutional_layers) + 1

        # convolution blocks
        modules_list = [
            nn.Conv2d(  in_channels=1, 
                        out_channels=self.convolution_channels, 
                        kernel_size=(1, 2), 
                        stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.convolution_channels),
        ]
        for _ in range(inner_convolutional_layers):
            modules_list.append(nn.Conv2d(  in_channels=self.convolution_channels, 
                                            out_channels=self.convolution_channels, 
                                            kernel_size=(self.smoothing_kernel_size, 1),
                                            stride=(1, 1)))
            modules_list.append(nn.LeakyReLU(negative_slope=0.01))
            modules_list.append(nn.BatchNorm2d(self.convolution_channels))
        self.conv1 = nn.Sequential(*modules_list)
        
        modules_list = [
            nn.Conv2d(  in_channels=self.convolution_channels, 
                        out_channels=self.convolution_channels, 
                        kernel_size=(1, 2), 
                        stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.convolution_channels),
        ]
        for _ in range(inner_convolutional_layers):
            modules_list.append(nn.Conv2d(  in_channels=self.convolution_channels, 
                                            out_channels=self.convolution_channels, 
                                            kernel_size=(self.smoothing_kernel_size, 1),
                                            stride=(1, 1)))
            modules_list.append(nn.LeakyReLU(negative_slope=0.01))
            modules_list.append(nn.BatchNorm2d(self.convolution_channels))
        self.conv2 = nn.Sequential(*modules_list)

        modules_list = [
            nn.Conv2d(  in_channels=self.convolution_channels, 
                        out_channels=self.convolution_channels, 
                        kernel_size=(1, 10), 
                        stride=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.convolution_channels),
        ]
        for _ in range(inner_convolutional_layers):
            modules_list.append(nn.Conv2d(  in_channels=self.convolution_channels, 
                                            out_channels=self.convolution_channels, 
                                            kernel_size=(self.smoothing_kernel_size, 1), 
                                            stride=(1, 1)))
            modules_list.append(nn.LeakyReLU(negative_slope=0.01))
            modules_list.append(nn.BatchNorm2d(self.convolution_channels))
        self.conv3 = nn.Sequential(*modules_list)

        # Inception modules: smoothing the time series with different scales.
        # A large kernel size is used to capture a global distribution of the image 
        # while a small kernel size is used to capture more local information.
        # Inception network architecture makes it possible to use filters of multiple 
        # sizes without increasing the depth of the network. The different filters 
        # are added parallelly instead of being fully connected one after the other.
        # The following scheme is the same as the one of the original paper of 
        # googleLeNet: https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf .
        self.inception_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.convolution_channels, out_channels=self.inception_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.inception_channels),
            nn.Conv2d(in_channels=self.inception_channels, out_channels=self.inception_channels, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.inception_channels),
        )
        self.inception_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.convolution_channels, out_channels=self.inception_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.inception_channels),
            nn.Conv2d(in_channels=self.inception_channels, out_channels=self.inception_channels, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.inception_channels),
        )
        self.inception_3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=self.convolution_channels, out_channels=self.inception_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.inception_channels),
        )
        self.inception_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.convolution_channels, out_channels=self.inception_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.inception_channels),
        )

        # RNN or lstm layer(s)
        self.dropout = nn.Dropout(0.1)
        self.gru = nn.GRU(input_size=self.inception_channels*4, hidden_size=self.inception_channels, num_layers=elf.lstm_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size=self.inception_channels*4, hidden_size=self.inception_channels, num_layers=self.lstm_layers, batch_first=True)
        self.linear_out = nn.Linear(self.inception_channels, self.output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inception_1 = self.inception_1(x)
        x_inception_2 = self.inception_2(x)
        x_inception_3 = self.inception_3(x)
        x_inception_4 = self.inception_4(x)

        x = torch.cat((x_inception_1, x_inception_2, x_inception_3, x_inception_4), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x = self.dropout(x)

        h0 = torch.zeros(self.lstm_layers, x.size(0), self.inception_channels).to(self.device)
        c0 = torch.zeros(self.lstm_layers, x.size(0), self.inception_channels).to(self.device)
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.linear_out(x)
        if self.regression:
            #x = torch.squeeze(x)
            return x
        else:
            forecast_y = torch.softmax(x, dim=1)
            return forecast_y
        
        # fixed params sizes
        #torch.Size([32, 1, 100, 40])
        #torch.Size([32, 32, 94, 20])
        #torch.Size([32, 32, 88, 10])
        #torch.Size([32, 32, 82, 1])
        #torch.Size([32, 64, 82, 1])
        #torch.Size([32, 64, 82, 1])
        #torch.Size([32, 64, 82, 1])
        #torch.Size([32, 192, 82, 1])
        #torch.Size([32, 82, 192])
        #torch.Size([32, 64])
        #torch.Size([32, 1])

        ## variable params sizes
        #torch.Size([32, 1, 100, 40])
        #torch.Size([32, 32, 94, 20])
        #torch.Size([32, 32, 88, 10])
        #torch.Size([32, 32, 82, 1])
        #torch.Size([32, 64, 82, 1])
        #torch.Size([32, 64, 82, 1])
        #torch.Size([32, 64, 82, 1])
        #torch.Size([32, 64, 82, 1])
        #torch.Size([32, 256, 82, 1])
        #torch.Size([32, 82, 256])
        #torch.Size([32, 64])
        #torch.Size([32, 1])



class Astice(nn.Module):
    
    def __init__(
        self,
        input_size  : int = 1,
        output_size : int = 1,
        hidden_size : int = 2,
        lstm_size : int = 2,
        lstm_layers : int = 1,
        smooth_conv_channels : int = 1,
    ):
        super(Astice, self).__init__()
        # Save model hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.smooth_conv_channels = smooth_conv_channels
        
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
        self.fcnn_input_size = lstm_size * lstm_layers
        self.fcnn_2 = nn.Sequential(
            nn.Linear(self.fcnn_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
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
        out = self.fcnn_2(x)
        
        return out.unsqueeze(1)
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
