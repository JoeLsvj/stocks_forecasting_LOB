import torch
import torch.nn as nn


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
