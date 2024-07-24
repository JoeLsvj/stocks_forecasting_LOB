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
        return predictions


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
