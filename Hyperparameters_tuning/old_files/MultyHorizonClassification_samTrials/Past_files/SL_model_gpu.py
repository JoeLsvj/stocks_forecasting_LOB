# Libs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# --------- Seq2Seq Model ------------



class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(32, 32, (4, 1), padding=(0, 0))  # Adjust padding here
        self.conv3 = nn.Conv2d(32, 32, (4, 1), padding=(0, 0))  # Adjust padding here
        self.conv4 = nn.Conv2d(32, 32, (1, 2), stride=(1, 2))
        self.conv5 = nn.Conv2d(32, 32, (4, 1), padding=(0, 0))  # Adjust padding here
        self.conv6 = nn.Conv2d(32, 32, (4, 1), padding=(0, 0))  # Adjust padding here
        self.conv7 = nn.Conv2d(32, 32, (1, 10))
        self.conv8 = nn.Conv2d(32, 32, (4, 1), padding=(0, 0))  # Adjust padding here
        self.conv9 = nn.Conv2d(32, 32, (4, 1), padding=(0, 0))  # Adjust padding here
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))
        x = self.leaky_relu(self.conv6(x))
        x = self.leaky_relu(self.conv7(x))
        x = self.leaky_relu(self.conv8(x))
        x = self.leaky_relu(self.conv9(x))
        return x

class InceptionModule(nn.Module):
    def __init__(self):
        super(InceptionModule, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, (1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(64, 64, (3, 1), padding=(1, 0))  # Adjust padding based on output size
        self.conv3 = nn.Conv2d(64, 64, (5, 1), padding=(2, 0))  # Adjust padding based on output size
        self.conv4 = nn.Conv2d(32, 64, (1, 1), padding=(0, 0))
        self.max_pool = nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0))  # Adjust padding based on output size
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        branch1 = self.leaky_relu(self.conv1(x))
        branch1 = self.leaky_relu(self.conv2(branch1))

        branch2 = self.leaky_relu(self.conv1(x))
        branch2 = self.leaky_relu(self.conv3(branch2))

        branch3 = self.max_pool(x)
        branch3 = self.leaky_relu(self.conv4(branch3))

        # Check and adjust tensor sizes if necessary
        # You may need to adjust tensor sizes using slicing or other operations
        
        outputs = torch.cat([branch1, branch2, branch3], 1)
        return outputs


class Seq2SeqModel(nn.Module):
    def __init__(self, latent_dim):
        super(Seq2SeqModel, self).__init__()
        self.encoder_cnn = EncoderCNN()
        self.inception = InceptionModule()
        self.encoder_lstm = nn.LSTM(192, latent_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(latent_dim + 3, latent_dim, batch_first=True)
        self.dense = nn.Linear(latent_dim, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_train, decoder_inputs):
        # Encoder CNN
        x = self.encoder_cnn(input_train)
        x = self.inception(x)
        
        # Print or log x.size() here to check the shape
        
        # Reshape for LSTM
        batch_size, channels, time_steps, features = x.size()
        x = x.reshape(batch_size, time_steps, channels * features)  # Adjust reshape as needed
        
        # Encoder LSTM
        encoder_outputs, (state_h, state_c) = self.encoder_lstm(x)
        states = (state_h, state_c)
        
        # Decoder
        all_outputs = []
        decoder_input = torch.cat([decoder_inputs, encoder_outputs[:, :1, :]], 2)
        
        for _ in range(5):
            decoder_output, states = self.decoder_lstm(decoder_input, states)
            output = self.dense(decoder_output)
            output = self.softmax(output)
            all_outputs.append(output)
            decoder_input = torch.cat([output, encoder_outputs[:, :1, :]], 2)
        
        decoder_outputs = torch.cat(all_outputs, 1)
        return decoder_outputs







# --------- Attention Model ------------

class Attention(nn.Module):
    def __init__(self, latent_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(latent_dim, latent_dim)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights
        attn_weights = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1)
        # Calculate the context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Seq2SeqAttentionModel(nn.Module):
    def __init__(self, latent_dim):
        super(Seq2SeqAttentionModel, self).__init__()
        self.encoder_cnn = EncoderCNN()
        self.inception = InceptionModule()
        self.encoder_lstm = nn.LSTM(192, latent_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(latent_dim + 3, latent_dim, batch_first=True)
        self.attention = Attention(latent_dim)
        self.dense = nn.Linear(latent_dim, 3)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm1d(latent_dim)

    def forward(self, input_train, decoder_inputs):
        # Encoder CNN
        x = self.encoder_cnn(input_train)
        x = self.inception(x)
        
        # Reshape for LSTM
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size, 50, -1)  # Reshape to (batch_size, time_steps, features)
        
        # Encoder LSTM
        encoder_outputs, (state_h, state_c) = self.encoder_lstm(x)
        
        # Decoder
        all_outputs = []
        all_attention = []
        decoder_input = torch.cat([decoder_inputs, state_h.permute(1, 0, 2)], 2)
        
        for _ in range(5):
            decoder_output, (state_h, state_c) = self.decoder_lstm(decoder_input, (state_h, state_c))
            context, attn_weights = self.attention(decoder_output, encoder_outputs)
            context = self.bn(context)
            decoder_combined_context = torch.cat([context.unsqueeze(1), decoder_output], dim=2)
            output = self.dense(decoder_combined_context)
            output = self.softmax(output)
            all_outputs.append(output)
            all_attention.append(attn_weights)
            decoder_input = torch.cat([output, context.unsqueeze(1)], 2)
        
        decoder_outputs = torch.cat(all_outputs, 1)
        return decoder_outputs, all_attention





