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
        return predictions
        #if (self.output_size==1):	return predictions[:,-1]
        #else:						return predictions


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
        return predictions
        #if (self.output_size==1):	return predictions[:,-1]
        #else:						return predictions


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

        # version 1 -> use the final encoder state
        # dec_input -> [num_layers+1, batch, encoder_hidden_size] +1 for the context vector
        #dec_input = torch.cat([encoder_h_c, context_vector.unsqueeze(0)], dim=0)
        ## dec_input -> [batch, num_layers+1, encoder_hidden_size]
        #dec_input = dec_input.permute(1, 0, 2)

        # version 2 -> use the whole encoder output
        # dec_input -> [batch, lookback_period+1, encoder_hidden_size]
        dec_input = torch.cat([encoder_outputs, context_vector.unsqueeze(1)], dim=1)

        # lstm decoder layer
        output, (decoder_h_c, decoder_c_n) = self.decoder(dec_input) # encoder_h_c, encoder_c_n
        x = decoder_h_c.permute(1, 0, 2).reshape(batchsize, -1)
        # final linear layer
        x = self.dropout(x)
        predictions = self.linear_out(x)
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


# mixed convolutional and LSTM architecture, with general structure
class CNN_LSTM(nn.Module):
    # Classe di costruzione della rete
    def __init__(self, H, W, min_size, ch_in_0, ch_out_0, f_1, f_2, c_layers_up, c_layers_down,
                c_size, c_stride, c_padd, c_num, p_type, p_size, p_stride, p_padd, fully_layers,
                hidden_layer_size=64, num_layers=2, output_lstm_size=100, dropout=0.4):

        super().__init__()

        ch_in = ch_in_0; ch_out = ch_out_0
        H_0 = H; W_0 = W

        # flag per interrompere la costruzione della rete e non scendere sotto min_size
        self.stop = False
        def size_check(H, W, padd, ksize, stride):
            # funzione che interrompe la costruzione della convolutional prima
            # di ridurre la dimensione dell'immagine oltre min_size
            dil = (1,1) # dilation
            H_new = int((H + 2*padd[0] - dil[0]*(ksize[0] - 1) - 1)/stride[0] + 1)
            W_new = int((W + 2*padd[1] - dil[1]*(ksize[1] - 1) - 1)/stride[1] + 1)
            if ((H_new < min_size) or (W_new < min_size)):
                self.stop = True
                print("\nRaggiunta dimensione minima dell'immagine:  ", H, "x", W)
                return H, W
            return H_new, W_new

        # formulas to define the trend to increment/decrement the channels in the convolutional part of the network
        formula_up = lambda x: eval(f_1)
        formula_down = lambda x: eval(f_2)

        # selecting the pooling type:
        if p_type == "max":
            Pool2d = nn.MaxPool2d(p_size, p_stride, p_padd)
        elif p_type == "avg":
            Pool2d = nn.AvgPool2d(p_size, p_stride, p_padd)
        else:
            raise ValueError("Type di Pool2d non valido, scegliere 'max' (default) o 'avg'")

        # features extraction -> encoding information
        self.encoder = nn.ModuleList([])
        # raising-channels layers:
        for i in range(0, c_layers_up):
            # conv2D layer that increment the channels:
            H, W = size_check(H, W, c_padd, c_size, c_stride)
            if self.stop:
                break
            self.encoder.append(nn.Conv2d(ch_in, ch_out, kernel_size=c_size, stride=c_stride, padding=c_padd))
            # inner conv2D layers that do not increase channels, but still reduce the image
            for _ in range(1, c_num):
                H, W = size_check(H, W, c_padd, c_size, c_stride)
                if self.stop:
                    break
                self.encoder.append(nn.Conv2d(ch_out, ch_out, kernel_size=c_size, stride=c_stride, padding=c_padd))
            if self.stop:
                break
            # batch-normalization and activation function
            self.encoder.append(nn.BatchNorm2d(ch_out))
            self.encoder.append(nn.ReLU())
            # pooling
            H, W = size_check(H, W, p_padd, p_size, p_stride)
            if self.stop:
                break
            self.encoder.append(Pool2d)
            ch_in = ch_out
            ch_out = ch_out_0*formula_up(i+1)
        # decreasing-channels layers: optional
        for i in range(0, c_layers_down):
            # conv2D layer that reduce the channels:
            ch_out = int(ch_in*formula_down(i+1))
            H, W = size_check(H, W, c_padd, c_size, c_stride)
            if self.stop:
                break
            self.encoder.append(nn.Conv2d(ch_in, ch_out, kernel_size=c_size, stride=c_stride, padding=c_padd))
            self.encoder.append(nn.BatchNorm2d(ch_out))
            self.encoder.append(nn.ReLU())
            H, W = size_check(H, W, p_padd, p_size, p_stride)
            if self.stop:
                break
            self.encoder.append(Pool2d)
            ch_in = ch_out
        # Now the tensor size is [batch, channels, H, W] -> H, W reduced by the convolutional phase.

        nz = 100
        # linear layers to compute the latent vector
        self.enc_bottleneck = nn.Sequential(
            nn.Linear(ch_in*H*W, ch_in*H*W),
            nn.BatchNorm1d(ch_in*H*W),
            nn.ReLU(),
            nn.Linear(ch_in*H*W, nz)
        )
        # reconstruction from the latent vector
        self.dec_bottleneck = nn.Sequential(
            nn.Linear(nz, ch_in*H*W),
            nn.BatchNorm1d(ch_in*H*W),
            nn.ReLU(),
            nn.Linear(ch_in*H*W, ch_in*H*W)
        )

        # reconstruction -> decoding
        # back the original H, W with transposedconv2D layers -> channels back to 1
        ch_in_0 = ch_in
        self.decoder = nn.ModuleList([])
        for i in range(0, c_layers_up - c_layers_down -1):
            ch_out = int(ch_in_0*formula_down(i+1)) # not correct: divide the cases
            self.decoder.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size=c_size, stride=c_stride, padding=c_padd))
            self.decoder.append(nn.BatchNorm2d(ch_out))
            self.decoder.append(nn.ReLU())
            self.decoder.append(Pool2d)
            ch_in = ch_out
        # last layer
        self.decoder.append(nn.ConvTranspose2d(ch_in, 1, kernel_size=c_size, stride=c_stride, padding=c_padd))
        self.decoder.append(nn.BatchNorm2d(1))
        self.decoder.append(nn.ReLU())
        self.decoder.append(nn.AdaptiveAvgPool2d((H_0, W_0)))

        # LSTM part
        self.hidden_layer_size = hidden_layer_size
        self.linear_1 = nn.Linear(W_0, hidden_layer_size)

        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size = self.hidden_layer_size, num_layers = num_layers, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_lstm_size)
        self.init_weights()

        # building the deep fully connected final network
        self.fully_connected = nn.ModuleList([])
        f_in = output_lstm_size
        for i in range(0, len(fully_layers)):
            f_out = fully_layers[i][0]
            f_layers = fully_layers[i][1]
            self.fully_connected.append(nn.Linear(f_in, f_out))
            self.fully_connected.append(nn.BatchNorm1d(f_out))
            self.fully_connected.append(nn.ReLU())
            self.fully_connected.append(nn.Dropout(dropout))
            for _ in range(1, f_layers):
                self.fully_connected.append(nn.Linear(f_out, f_out))
                self.fully_connected.append(nn.BatchNorm1d(f_out))
                self.fully_connected.append(nn.ReLU())
                self.fully_connected.append(nn.Dropout(dropout))
            f_in = f_out
        self.fully_connected.append(nn.Linear(f_out, 1))

    def forward(self, x):
        for i in enumerate(self.encoder):
            x = self.encoder[i[0]](x)

        x_sizes = [x.size(0), x.size(1), x.size(2), x.size(3)]
        x = x.view(-1, x_sizes[1]*x_sizes[2]*x_sizes[3])
        # l'output del modulo successivo è il vettore latente (z)
        z = self.enc_bottleneck(x)
        x = self.dec_bottleneck(z)
        x = x.view(-1, x_sizes[1], x_sizes[2], x_sizes[3])	
        # x = self.dec_pool(x)

        for i in enumerate(self.decoder):
            x = self.decoder[i[0]](x)
        
        x = torch.squeeze(x, 1) # delete the channel dimension since is = 1 if all have worked
        batchsize = x.shape[0]
        # linear lstm layer 1
        x = self.linear_1(x)
        x = self.relu(x)
        # LSTM layer
        # The input x has a shape [batch, timestamp, n features = input features]
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        # layer 2
        x = self.dropout(x)
        x = self.linear_2(x)

        x = x.reshape(x.size(0), -1)
        for i in enumerate(self.fully_connected):
            x = self.fully_connected[i[0]](x)
        # If the following is decommented, we have not to unsqueeze the labels in the main code.
        #out = torch.squeeze(x)
        return x[:, -1] #out

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)


# full CNN approach
class DynamicCNN(nn.Module):
    # Classe di costruzione della rete
    def __init__(self, H, W, min_size, ch_in_0, ch_out_0, f_1, f_2, c_layers_up, c_layers_down, c_size, c_stride, c_padd, c_num, p_type, p_size, p_stride, p_padd, fully_layers):
        super(DynamicCNN, self).__init__()

        ch_in = ch_in_0
        ch_out = ch_out_0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'DynamicCNN'

        self.stop = False
        # flag per interrompere la costruzione della rete e non scendere sotto min_size

        def size_check(H, W, padd, ksize, stride):
            # funzione che interrompe la costruzione della convolutional prima
            # di ridurre la dimensione dell'immagine oltre min_size
            dil = (1,1) # dilation
            H_new = int((H + 2*padd[0] - dil[0]*(ksize[0] - 1) - 1)/stride[0] + 1)
            W_new = int((W + 2*padd[1] - dil[1]*(ksize[1] - 1) - 1)/stride[1] + 1)
            if ((H_new < min_size) or (W_new < min_size)):
                self.stop = True
                print("\nRaggiunta dimensione minima dell'immagine:  ", H, "x", W)
                return H, W
            return H_new, W_new

        # formule di definizione del numero di canali delle conv2d
        formula_up = lambda x: eval(f_1)
        formula_down = lambda x: eval(f_2)

        # seleziono il tipo di pooling
        if p_type == "max":
            Pool2d = nn.MaxPool2d(p_size, p_stride, p_padd)
        elif p_type == "avg":
            Pool2d = nn.AvgPool2d(p_size, p_stride, p_padd)
        else:
            raise ValueError("Type di Pool2d non valido, scegliere 'max' (default) o 'avg'")

        # costruzione features extraction
        # parte di conv che aumenta i canali
        self.conv = nn.ModuleList([])
        for i in range(0, c_layers_up):
            H, W = size_check(H, W, c_padd, c_size, c_stride)
            if self.stop:
                break
            self.conv.append(nn.Conv2d(ch_in, ch_out, kernel_size=c_size, stride=c_stride, padding=c_padd))
            for _ in range(1, c_num):
                H, W = size_check(H, W, c_padd, c_size, c_stride)
                if self.stop:
                    break
                self.conv.append(nn.Conv2d(ch_out, ch_out, kernel_size=c_size, stride=c_stride, padding=c_padd))
            if self.stop:
                break
            self.conv.append(nn.BatchNorm2d(ch_out))
            self.conv.append(nn.ReLU())
            H, W = size_check(H, W, p_padd, p_size, p_stride)
            if self.stop:
                break
            self.conv.append(Pool2d)
            ch_in = ch_out
            ch_out_up = ch_out
            ch_out = ch_out_0*formula_up(i+1)

        # parte di conv che riduce i canali
        for i in range(0, c_layers_down):
            ch_out = int(ch_out_up*formula_down(-i-1))
            H, W = size_check(H, W, c_padd, c_size, c_stride)
            if self.stop:
                break
            self.conv.append(nn.Conv2d(ch_in, ch_out, kernel_size=c_size, stride=c_stride, padding=c_padd))
            self.conv.append(nn.BatchNorm2d(ch_out))
            self.conv.append(nn.ReLU())
            H, W = size_check(H, W, p_padd, p_size, p_stride)
            if self.stop:
                break
            self.conv.append(Pool2d)
            ch_in = ch_out

        # costruzione fully connected (classifier)
        self.fully_connected = nn.ModuleList([])
        f_in = ch_out*W*H
        for i in range(0, len(fully_layers)):
            f_out = fully_layers[i][0]
            f_layers = fully_layers[i][1]
            self.fully_connected.append(nn.Linear(f_in, f_out))
            self.fully_connected.append(nn.BatchNorm1d(f_out))
            self.fully_connected.append(nn.ReLU())
            self.fully_connected.append(nn.Dropout(p=0.8))
            for _ in range(1, f_layers):
                self.fully_connected.append(nn.Linear(f_out, f_out))
                self.fully_connected.append(nn.BatchNorm1d(f_out))
                self.fully_connected.append(nn.ReLU())
                self.fully_connected.append(nn.Dropout(p=0.8))
            f_in = f_out
        self.fully_connected.append(nn.Linear(f_out, 1))

    def forward(self, x):
        for i in enumerate(self.conv):
            print(x.shape)
            x = self.conv[i[0]](x)
        #print("\n")
        print(x.shape)
        x = x.reshape(x.size(0), -1)
        print(x.shape)
        for i in enumerate(self.fully_connected):
            x = self.fully_connected[i[0]](x)
            print(x.shape)
        #out = x
        # If the following is decommented, we have not to unsqueeze the labels in the main code.
        # In the contrary (used the line before), unsqueeze the label dimension on axis 1, to obtain
        # a torch.tensor([16,1]) form a torch.tensor([16]).
        out = torch.squeeze(x)
        return out