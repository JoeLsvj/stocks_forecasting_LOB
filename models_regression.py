import torch
import torch.nn as nn
import torch.nn.functional as F

# simple LSTM model
class LSTM1(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, dropout=0.2, output_size=1):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'LSTM1'

        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.linear_1 = nn.Linear(input_size, self.hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(self.hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(self.num_layers*self.hidden_layer_size, output_size)
        
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
        return predictions.unsqueeze(1)


# double LSTM model
class LSTM2(nn.Module):
    def __init__(self, input_size=1, amplification_size=256, hidden_layer_size=32, num_layers=2, dropout=0.2, output_size=1):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'LSTM2'

        self.amplification_size = amplification_size
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.linear_1 = nn.Linear(input_size, self.amplification_size)
        self.relu = nn.ReLU()
        self.Dropout = nn.Dropout(self.dropout)
        self.lstm_1 = nn.LSTM(self.amplification_size, hidden_size=self.hidden_layer_size, num_layers=self.num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(self.hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers =self.num_layers, batch_first=True)
        self.linear_2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.linear_3 = nn.Linear(self.num_layers*self.hidden_layer_size, output_size)
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm_1.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
        for name, param in self.lstm_2.named_parameters():
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
        lstm_out, (h_n, c_n) = self.lstm_1(x)
        # linear layer 2
        x = self.linear_2(lstm_out)
        x = self.relu(x)
        x = self.Dropout(x)
        lstm_out, (h_n, c_n) = self.lstm_2(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.Dropout(x)
        predictions = self.linear_3(x)
        return predictions.unsqueeze(1)


# encoder-decoder LSTM with attention mechanism
class LSTMWithAttention(nn.Module):

    def __init__(self, input_size=1, 
                encoder_hidden_size=32, 
                decoder_hidden_size=32,
                num_layers = 2,
                dropout=0.2,
                output_size=1,
            ):
        super(LSTMWithAttention, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'LSTMAttention'

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        self.relu = nn.ReLU()
        #self.linear_out = nn.Linear(decoder_layers*decoder_hidden_size, output_size)
        
        # Encoder
        self.encoder = nn.LSTM(input_size, encoder_hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        # Attention mechanism
        self.attn_layer = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size)
        # generate the learnable parameters for attention scores:
        self.v = nn.Parameter(torch.rand(decoder_hidden_size))
        
        # Decoder
        self.decoder = nn.LSTM(encoder_hidden_size + 1, decoder_hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fcnn = nn.Linear(decoder_hidden_size, 1)

        self.init_weights()


    def forward(self, x):#, target, teacher_forcing_ratio=0.5):
        # x:        [batch_size, lookback_period, input_size]
        # target:   [batch_size, 1, output_size]
        batch_size  = x.size(0)
        #trg_seq_len = target.size(1)
        #output_size = target.size(2)
        
        # Placeholder for final decoder's outputs
        outputs = torch.zeros(batch_size, 1, self.output_size).to(x.device)
        # Encoder: h_n = hidden state, c_n = cell state
        encoder_outputs, (h_n, c_n) = self.encoder(x)
        # First input to the decoder
        #tf_input = target[:, :, 0].unsqueeze(2)                                         # [batch_size, 1, output_size]
        tf_input = torch.zeros(batch_size, 1, 1).to(x.device)
        for t in range(1, self.output_size):
            attn_weights = self.compute_attention_weights(encoder_outputs, h_n[-1])     # [batch_size, lookback_period]
            context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)      # [batch_size, 1, encoder_hidden_size]
            #print(context_vector.shape)
            # Decoder input: concatenate input and context_vector
            decoder_input = torch.cat((tf_input, context_vector), dim=2)                # [batch_size, 1, encoder_hidden_size + output_size]
            #print(decoder_input.shape)
            output, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            #print(output.shape)
            prediction = self.fcnn(output.squeeze(1))                                   # [batch_size, output_size]
            #print(prediction.shape)
            outputs[:, :, t] = prediction
            
            #teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            tf_input = prediction.unsqueeze(1) #target[:, :, t].unsqueeze(1) if teacher_force else prediction.unsqueeze(1)
            #print(tf_input.shape)
        
        return outputs
    

    def compute_attention_weights(self, encoder_outputs, h_n):
        # encoder_outputs: [batch_size, seq_len, encoder_hidden_size]
        # h_n: [batch_size, decoder_hidden_size]
        seq_len = encoder_outputs.size(1)
        
        # Repeat h_n state seq_len times
        h_n = h_n.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate h_n and encoder outputs
        intermediate_attn_scores = torch.tanh(self.attn_layer(torch.cat((h_n, encoder_outputs), dim=2)))    # [batch_size, seq_len, decoder_hidden_size]
        intermediate_attn_scores = intermediate_attn_scores.permute(0, 2, 1)                                # [batch_size, decoder_hidden_size, seq_len]
        
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)          # [batch_size, 1, decoder_hidden_size]
        attn_scores = torch.bmm(v, intermediate_attn_scores).squeeze(1)     # [batch_size, seq_len]
        
        return F.softmax(attn_scores, dim=1)


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
                    vertical_convolutional_layers = 2,
                    vertical_kernel_size = 4,
                    smoothing:bool = False,
                    convolution_channels = 32,
                    inception_channels = 64,
                    lstm_layers = 2,
                    latent_size = 64,
                    dropout = 0.2,
                    output_size = 1,
                ):
        super(LobCNN, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'LobCNN'

        self.vertical_convolutional_layers = vertical_convolutional_layers
        self.vertical_kernel_size = vertical_kernel_size
        self.smoothing = smoothing
        self.convolution_channels = convolution_channels
        self.inception_channels = inception_channels
        self.lstm_layers = lstm_layers
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        # The value of the smoothing can be computed with (if smoothing == True):
        # vertical_kernel_size = int(smoothing_percentage * lookback_period/vertical_convolutional_layers) + 1

        # convolution blocks
        if self.smoothing:
            padding_value = (0, 0)
        else: padding_value = 'same'

        modules_list = [
            nn.Conv2d(  in_channels=1, 
                        out_channels=self.convolution_channels, 
                        kernel_size=(1, 2), 
                        stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.convolution_channels),
        ]
        for _ in range(self.vertical_convolutional_layers):
            modules_list.append(nn.Conv2d(  in_channels=self.convolution_channels,
                                            out_channels=self.convolution_channels,
                                            kernel_size=(self.vertical_kernel_size, 1),
                                            stride=(1, 1),
                                            padding=padding_value
                                            ))
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
        for _ in range(self.vertical_convolutional_layers):
            modules_list.append(nn.Conv2d(  in_channels=self.convolution_channels,
                                            out_channels=self.convolution_channels,
                                            kernel_size=(self.vertical_kernel_size, 1),
                                            stride=(1, 1),
                                            padding=padding_value
                                            ))
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
        for _ in range(self.vertical_convolutional_layers):
            modules_list.append(nn.Conv2d(  in_channels=self.convolution_channels, 
                                            out_channels=self.convolution_channels, 
                                            kernel_size=(self.vertical_kernel_size, 1), 
                                            stride=(1, 1),
                                            padding=padding_value
                                            ))
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

        # RNN or LSTM layer(s)
        self.Dropout = nn.Dropout(self.dropout)
        #self.encoder_gru = nn.GRU(input_size=self.inception_channels*4, hidden_size=self.inception_channels, num_layers=self.lstm_layers, batch_first=True)
        self.encoder_lstm = nn.LSTM(input_size=self.inception_channels*4, hidden_size=self.latent_size, num_layers=self.lstm_layers, batch_first=True)
        
        # Output FCNN
        self.fcnn_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size, self.latent_size),
            nn.LeakyReLU(),
            nn.Linear(self.latent_size, self.output_size)
        )

        self.init_weights()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inception_1 = self.inception_1(x)
        x_inception_2 = self.inception_2(x)
        x_inception_3 = self.inception_3(x)
        x_inception_4 = self.inception_4(x)

        x = torch.cat((x_inception_1, x_inception_2, x_inception_3, x_inception_4), dim=1)
        x = x.permute(0, 2, 1, 3)#.squeeze(3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x = self.Dropout(x)

        h0 = torch.zeros(self.lstm_layers, x.size(0), self.latent_size).to(self.device)
        c0 = torch.zeros(self.lstm_layers, x.size(0), self.latent_size).to(self.device)
        x, _ = self.encoder_lstm(x, (h0, c0))
        x = x[:, -1, :]

        x = self.fcnn_out(x).unsqueeze(1).unsqueeze(1)
        return x

    def init_weights(self):
        for name, param in self.encoder_lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)



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
        self.name = 'Astice'
        self.device = self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        self.init_weights()
        
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



# with decoders.
class LobCNN_encoderdecoder(nn.Module):
    def __init__(   self,
                    vertical_convolutional_layers = 2,
                    vertical_kernel_size = 2,
                    smoothing:bool = False,
                    convolution_channels = 32,
                    inception_channels = 64,
                    lstm_layers = 2,
                    latent_size = 64,
                    output_size = 1,
                ):
        super(LobCNN_encoderdecoder, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'LobCNN_encoderdecoder'

        self.vertical_convolutional_layers = vertical_convolutional_layers
        self.vertical_kernel_size = vertical_kernel_size
        self.smoothing = smoothing
        self.convolution_channels = convolution_channels
        self.inception_channels = inception_channels
        self.lstm_layers = lstm_layers
        self.latent_size = latent_size
        self.output_size = output_size
        # The value of the smoothing can be computed with (if smoothing == True):
        # vertical_kernel_size = int(smoothing_percentage * lookback_period/vertical_convolutional_layers) + 1

        # convolution blocks
        if self.smoothing:
            padding_value = (0, 0)
        else: padding_value = 'same'

        modules_list = [
            nn.Conv2d(  in_channels=1, 
                        out_channels=self.convolution_channels, 
                        kernel_size=(1, 2), 
                        stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.convolution_channels),
        ]
        for _ in range(self.vertical_convolutional_layers):
            modules_list.append(nn.Conv2d(  in_channels=self.convolution_channels,
                                            out_channels=self.convolution_channels,
                                            kernel_size=(self.vertical_kernel_size, 1),
                                            stride=(1, 1),
                                            padding=padding_value
                                            ))
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
        for _ in range(self.vertical_convolutional_layers):
            modules_list.append(nn.Conv2d(  in_channels=self.convolution_channels,
                                            out_channels=self.convolution_channels,
                                            kernel_size=(self.vertical_kernel_size, 1),
                                            stride=(1, 1),
                                            padding=padding_value
                                            ))
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
        for _ in range(self.vertical_convolutional_layers):
            modules_list.append(nn.Conv2d(  in_channels=self.convolution_channels, 
                                            out_channels=self.convolution_channels, 
                                            kernel_size=(self.vertical_kernel_size, 1), 
                                            stride=(1, 1),
                                            padding=padding_value
                                            ))
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

        # RNN or LSTM layer(s)
        self.dropout = nn.Dropout(0.2)
        #self.encoder_gru = nn.GRU(input_size=self.inception_channels*4, hidden_size=self.latent_size, num_layers=self.lstm_layers, batch_first=True)
        self.encoder_lstm = nn.LSTM(input_size=self.inception_channels*4, hidden_size=self.latent_size, num_layers=self.lstm_layers, batch_first=True)

        self.decoder_lstm = nn.LSTM(input_size=self.latent_size, hidden_size=self.latent_size, num_layers=self.lstm_layers, batch_first=True)
        #self.fcnn = nn.Linear(self.latent_size * 2, 3) # classification
        self.fcnn = nn.Linear(self.latent_size * 2, self.output_size) # regression
        self.softmax = nn.Softmax(dim=-1)
        self.batch_norm = nn.BatchNorm1d(self.latent_size)

        self.init_weights()

    
    def convolution(self, x):
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
        return x

    def encoder(self, x):
        h0 = torch.zeros(self.lstm_layers, x.size(0), self.latent_size).to(self.device)
        c0 = torch.zeros(self.lstm_layers, x.size(0), self.latent_size).to(self.device)
        encoder_output, (hn, cn) = self.encoder_lstm(x, (h0, c0))
        return encoder_output, (hn, cn)
    
    def decoder(self, decoder_input, encoder_output, states):
        # Decoder with attention
        all_outputs = []
        all_attention = []
        # Initial inputs and state for the decoder [32, 1, 1 or 2] or [32, 1, 128], state_h: [2, 32, 128]
        #decoder_input = torch.cat((decoder_input, state_h.permute(1,0,2)), dim=1)
        for _ in range(self.output_size):

            outputs, states = self.decoder_lstm(decoder_input, states)
            attention = torch.bmm(outputs, encoder_output.permute(0, 2, 1))
            attention = self.softmax(attention)
            context = torch.bmm(attention, encoder_output)
            context = self.batch_norm(context.permute(0, 2, 1)).permute(0, 2, 1)
            decoder_combined_context = torch.cat([context, outputs], dim=-1)
            outputs = self.fcnn(decoder_combined_context)
            all_outputs.append(outputs)
            all_attention.append(attention)
            decoder_input = torch.cat([outputs, context[:,:,:-1]], dim=-1)

        decoder_outputs = torch.cat(all_outputs, dim=1).squeeze(2)
        decoder_attention = torch.cat(all_attention, dim=1)
        return decoder_outputs, decoder_attention


    def forward(self, x):
        decoder_input = torch.zeros(x.shape[0], 1, self.latent_size).to(self.device)
        #decoder_input = torch.zeros(x.shape[0], 1, 1).to(self.device)

        x = self.convolution(x)
        encoder_output, states = self.encoder(x)
        decoder_outputs, _ = self.decoder(decoder_input, encoder_output, states)
        return decoder_outputs


    def init_weights(self):
        for name, param in self.encoder_lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
        for name, param in self.decoder_lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)


class LobCNN_encoderdecoder_light(nn.Module):
    def __init__(self, latent_size, output_size):
        super(LobCNN_encoderdecoder_light, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'LobCNN_encoderdecoder_light'

        self.latent_size = latent_size
        self.output_size = output_size
        
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same')
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same')
        
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same')
        self.conv6 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same')
        
        self.conv7 = nn.Conv2d(32, 32, kernel_size=(1, 10), stride=(1, 2))
        self.conv8 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same')
        self.conv9 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same')
        
        # Inception module
        self.conv10 = nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same')
        self.conv11 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding='same')
        
        self.conv12 = nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same')
        self.conv13 = nn.Conv2d(64, 64, kernel_size=(5, 1), padding='same')
        
        self.conv14 = nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same')
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1,0))
        
        # LSTM layers
        self.encoder_lstm = nn.LSTM(64*3, latent_size, num_layers=2, batch_first=True)
        self.decoder_lstm = nn.LSTM(1, latent_size, num_layers=2, batch_first=True)
        
        #self.dense = nn.Linear(latent_size * 2, 3) # classification
        self.dense = nn.Linear(latent_size * 2, output_size) # regression
        self.softmax = nn.Softmax(dim=-1)
        self.batch_norm = nn.BatchNorm1d(latent_size)

        self.init_weights()


    def convolution(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        
        x = F.leaky_relu(self.conv4(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv6(x), negative_slope=0.01)
        
        x = F.leaky_relu(self.conv7(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv8(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv9(x), negative_slope=0.01)

        # Inception module
        inception_1 = F.leaky_relu(self.conv10(x), negative_slope=0.01)
        inception_1 = F.leaky_relu(self.conv11(inception_1), negative_slope=0.01)
        
        inception_2 = F.leaky_relu(self.conv12(x), negative_slope=0.01)
        inception_2 = F.leaky_relu(self.conv13(inception_2), negative_slope=0.01)
        
        inception_3 = self.pool(x)
        inception_3 = F.leaky_relu(self.conv14(inception_3), negative_slope=0.01)
        
        convolution_output = torch.cat([inception_1, inception_2, inception_3], dim=1)
        convolution_output = convolution_output.permute(0, 2, 1, 3).reshape(convolution_output.size(0), -1, convolution_output.size(1))
        return convolution_output

    def encoder(self, convolution_output):
        # Encoder LSTM
        encoder_output, (state_h, state_c) = self.encoder_lstm(convolution_output)
        return encoder_output, (state_h, state_c)

    def decoder(self, decoder_input, encoder_output, states):
        # Decoder with attention
        all_outputs = []
        all_attention = []

        # Initial inputs and state for the decoder [32, 1, 1] or [32, 1, 128], state_h: [2, 32, 128]
        #decoder_input = torch.cat((decoder_input, state_h.permute(1,0,2)), dim=1)
        for _ in range(self.output_size):

            outputs, states = self.decoder_lstm(decoder_input, states)
            attention = torch.bmm(outputs, encoder_output.permute(0, 2, 1))
            attention = self.softmax(attention)
            context = torch.bmm(attention, encoder_output)
            context = self.batch_norm(context.permute(0, 2, 1)).permute(0, 2, 1)
            decoder_combined_context = torch.cat([context, outputs], dim=-1)
            outputs = self.dense(decoder_combined_context)
            all_outputs.append(outputs)
            all_attention.append(attention)
            decoder_input = torch.cat([outputs, context[:,:,1:]], dim=-1)

        decoder_outputs = torch.cat(all_outputs, dim=1).squeeze(2)
        decoder_attention = torch.cat(all_attention, dim=1)
        return decoder_outputs, decoder_attention


    def forward(self, x):
        #decoder_input = torch.zeros(x.shape[0], 1, self.latent_size).to(self.device)
        decoder_input = torch.zeros(x.shape[0], 1, 1).to(self.device)

        convolution_output = self.convolution(x)
        encoder_output, states = self.encoder(convolution_output)
        decoder_outputs, _ = self.decoder(decoder_input, encoder_output, states)
        return decoder_outputs


    def init_weights(self):
        for name, param in self.encoder_lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
        for name, param in self.decoder_lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)



class LSTMSeq2SeqAttention(nn.Module):
    # The attention mechanism implemented in the code is inspired by the work of Luong et al. (2015), 
    # often referred to as "Luong attention". The specific variant used here involves computing 
    # alignment scores based on the concatenation of the encoder's hidden states and the decoder's 
    # hidden state, followed by a linear transformation and a dot product with a learnable parameter vector. 
    # 
    # 1. **Intermediate alignment score**: $ \mathbf{e}_{ti} = \tanh(\mathbf{W}[\mathbf{h}_t; \mathbf{h}_i]) $
    # 2. **Scalar score**: $ a_{ti} = \mathbf{v}^\top \mathbf{e}_{ti} $
    # 3. **Attention weight**: $ \alpha_{ti} = \frac{\exp(a_{ti})}{\sum_{j} \exp(a_{tj})} $
    # 4. **Context vector**: $ \mathbf{c}_t = \sum_{i} \alpha_{ti} \mathbf{h}_i $
    def __init__(   self, 
                    input_size, 
                    output_size=1, 
                    encoder_hidden_size=64, 
                    decoder_hidden_size=64, 
                    num_layers=2, 
                    dropout=0.1
                ):

        super(LSTMSeq2SeqAttention, self).__init__()

        self.name = 'LSTM-seq-to-seq-wAttention'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Encoder
        self.encoder_hidden_size = encoder_hidden_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size, encoder_hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        # Attention mechanism
        self.attn_layer = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size)
        # generate the learnable parameters for attention scores:
        self.v = nn.Parameter(torch.rand(decoder_hidden_size))
        
        # Decoder
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder = nn.LSTM(encoder_hidden_size + output_size, decoder_hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fcnn = nn.Linear(decoder_hidden_size, output_size)

        self.init_weights()
        
    def forward(self, x, target, teacher_forcing_ratio=0.5):
        # x:        [batch_size, lookback_period, input_size]
        # target:   [batch_size, trg_seq_len, output_size]
        batch_size  = x.size(0)
        trg_seq_len = target.size(1)
        output_size = target.size(2)
        
        # Placeholder for final decoder's outputs
        outputs = torch.zeros(batch_size, trg_seq_len, output_size).to(x.device)
        # Encoder: h_n = hidden state, c_n = cell state
        encoder_outputs, (h_n, c_n) = self.encoder(x)
        # First input to the decoder
        tf_input = target[:, 0, :].unsqueeze(1)                                         # [batch_size, 1, output_size]
        
        for t in range(1, trg_seq_len):
            attn_weights = self.compute_attention_weights(encoder_outputs, h_n[-1])     # [batch_size, lookback_period]
            context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)      # [batch_size, 1, encoder_hidden_size]
            
            # Decoder input: concatenate input and context_vector
            decoder_input = torch.cat((tf_input, context_vector), dim=2)                # [batch_size, 1, encoder_hidden_size + output_size]
            output, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            
            prediction = self.fcnn(output.squeeze(1))                                   # [batch_size, output_size]
            outputs[:, t, :] = prediction
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            tf_input = target[:, t, :].unsqueeze(1) if teacher_force else prediction.unsqueeze(1)
        
        return outputs
    
    def compute_attention_weights(self, encoder_outputs, h_n):
        # encoder_outputs: [batch_size, seq_len, encoder_hidden_size]
        # h_n: [batch_size, decoder_hidden_size]
        seq_len = encoder_outputs.size(1)
        
        # Repeat h_n state seq_len times
        h_n = h_n.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate h_n and encoder outputs
        intermediate_attn_scores = torch.tanh(self.attn_layer(torch.cat((h_n, encoder_outputs), dim=2)))    # [batch_size, seq_len, decoder_hidden_size]
        intermediate_attn_scores = intermediate_attn_scores.permute(0, 2, 1)                                # [batch_size, decoder_hidden_size, seq_len]
        
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)          # [batch_size, 1, decoder_hidden_size]
        attn_scores = torch.bmm(v, intermediate_attn_scores).squeeze(1)     # [batch_size, seq_len]
        
        return F.softmax(attn_scores, dim=1)


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



class LobCNN_seq2seq_Attention(nn.Module):
    def __init__(   self,
                    vertical_convolutional_layers = 2,
                    vertical_kernel_size = 4,
                    smoothing:bool = False,
                    convolution_channels = 32,
                    inception_channels = 64,
                    lstm_layers = 2,
                    encoder_hidden_size = 64,
                    decoder_hidden_size = 64,
                    dropout = 0.2,
                    output_size = 1,
                ):
        super(LobCNN_seq2seq_Attention, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'LobCNN_seq2seq_Attention'

        self.vertical_convolutional_layers = vertical_convolutional_layers
        self.vertical_kernel_size = vertical_kernel_size
        self.smoothing = smoothing
        self.convolution_channels = convolution_channels
        self.inception_channels = inception_channels
        self.lstm_layers = lstm_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.dropout = dropout
        # The value of the smoothing can be computed with (if smoothing == True):
        # vertical_kernel_size = int(smoothing_percentage * lookback_period/vertical_convolutional_layers) + 1

        # convolution blocks
        if self.smoothing:
            padding_value = (0, 0)
        else: padding_value = 'same'

        modules_list = [
            nn.Conv2d(  in_channels=1, 
                        out_channels=self.convolution_channels, 
                        kernel_size=(1, 2), 
                        stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(self.convolution_channels),
        ]
        for _ in range(self.vertical_convolutional_layers):
            modules_list.append(nn.Conv2d(  in_channels=self.convolution_channels,
                                            out_channels=self.convolution_channels,
                                            kernel_size=(self.vertical_kernel_size, 1),
                                            stride=(1, 1),
                                            padding=padding_value
                                            ))
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
        for _ in range(self.vertical_convolutional_layers):
            modules_list.append(nn.Conv2d(  in_channels=self.convolution_channels,
                                            out_channels=self.convolution_channels,
                                            kernel_size=(self.vertical_kernel_size, 1),
                                            stride=(1, 1),
                                            padding=padding_value
                                            ))
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
        for _ in range(self.vertical_convolutional_layers):
            modules_list.append(nn.Conv2d(  in_channels=self.convolution_channels, 
                                            out_channels=self.convolution_channels, 
                                            kernel_size=(self.vertical_kernel_size, 1), 
                                            stride=(1, 1),
                                            padding=padding_value
                                            ))
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

        self.Dropout = nn.Dropout(self.dropout)
        
        # Encoder
        self.encoder_lstm = nn.LSTM(input_size=self.inception_channels*4,
                                    hidden_size=encoder_hidden_size,
                                    num_layers=lstm_layers,
                                    dropout=dropout, 
                                    batch_first=True
                            )

        # Attention mechanism
        self.attn_layer = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size)
        # generate the learnable parameters for attention scores:
        self.v = nn.Parameter(torch.rand(decoder_hidden_size))
        
        # Decoder
        self.decoder_lstm = nn.LSTM(input_size=encoder_hidden_size + output_size, 
                                    hidden_size=decoder_hidden_size, 
                                    num_layers=lstm_layers, 
                                    dropout=dropout, 
                                    batch_first=True
        )
        self.fcnn = nn.Linear(decoder_hidden_size, output_size)

        self.init_weights()


    def forward(self, x, target, teacher_forcing_ratio=0.5):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inception_1 = self.inception_1(x)
        x_inception_2 = self.inception_2(x)
        x_inception_3 = self.inception_3(x)
        x_inception_4 = self.inception_4(x)

        x = torch.cat((x_inception_1, x_inception_2, x_inception_3, x_inception_4), dim=1)
        x = x.permute(0, 2, 1, 3)#.squeeze(3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x = self.Dropout(x)

        print(x.shape)
        # x:        convolutional output: [batch, lookback_period, inception_channels*4 ~ 256]
        # target:   [batch_size, trg_seq_len, output_size]
        batch_size  = x.size(0)
        trg_seq_len = target.size(1)
        output_size = target.size(2)
        
        # Placeholder for final decoder's outputs
        outputs = torch.zeros(batch_size, trg_seq_len, output_size).to(x.device)
        # Encoder: h_n = hidden state, c_n = cell state
        #h0 = torch.zeros(self.lstm_layers, x.size(0), self.encoder_hidden_size).to(self.device)
        #c0 = torch.zeros(self.lstm_layers, x.size(0), self.encoder_hidden_size).to(self.device)
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(x)#, (h0, c0))
        # First input to the decoder
        tf_input = target[:, 0, :].unsqueeze(1)                                         # [batch_size, 1, output_size]
        
        for t in range(1, trg_seq_len):
            attn_weights = self.compute_attention_weights(encoder_outputs, h_n[-1])     # [batch_size, lookback_period]
            context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)      # [batch_size, 1, encoder_hidden_size]
            
            # Decoder input: concatenate input and context_vector
            decoder_input = torch.cat((tf_input, context_vector), dim=2)                # [batch_size, 1, encoder_hidden_size + output_size]
            output, (h_n, c_n) = self.decoder_lstm(decoder_input, (h_n, c_n))
            
            prediction = self.fcnn(output.squeeze(1))                                   # [batch_size, output_size]
            outputs[:, t, :] = prediction
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            tf_input = target[:, t, :].unsqueeze(1) if teacher_force else prediction.unsqueeze(1)
        
        return outputs.unsqueeze(-1)
    
    def compute_attention_weights(self, encoder_outputs, h_n):
        # encoder_outputs: [batch_size, seq_len, encoder_hidden_size]
        # h_n: [batch_size, decoder_hidden_size]
        seq_len = encoder_outputs.size(1)
        
        # Repeat h_n state seq_len times
        h_n = h_n.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate h_n and encoder outputs
        intermediate_attn_scores = torch.tanh(self.attn_layer(torch.cat((h_n, encoder_outputs), dim=2)))    # [batch_size, seq_len, decoder_hidden_size]
        intermediate_attn_scores = intermediate_attn_scores.permute(0, 2, 1)                                # [batch_size, decoder_hidden_size, seq_len]
        
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)          # [batch_size, 1, decoder_hidden_size]
        attn_scores = torch.bmm(v, intermediate_attn_scores).squeeze(1)     # [batch_size, seq_len]
        
        return F.softmax(attn_scores, dim=1)


    def init_weights(self):
        for name, param in self.encoder_lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
        for name, param in self.decoder_lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)