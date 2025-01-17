import torch.nn as nn
import torch
from torch.autograd import Variable
import functions
import math

class TemperatureLSTM(nn.Module):
    def __init__(self, lstmInputSize, lstmHiddenSize, num_layers, device):
        super(TemperatureLSTM, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.lstmHiddenSize = lstmHiddenSize
        self.lstmInputSize = lstmInputSize

        # LSTM for temporal processing
        self.lstm = nn.LSTM(input_size=self.lstmInputSize, hidden_size=self.lstmHiddenSize, 
                            num_layers=num_layers, batch_first=True)
        # Convolutional layer to capture spatial features post-LSTM
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, temperatures):
        # temperatures is a list of 4 tensors of shape (X, 50, 50)
        h = Variable(torch.zeros(self.num_layers, temperatures.size(0), self.lstmHiddenSize)).to(self.device)  # hidden state
        c = Variable(torch.zeros(self.num_layers, temperatures.size(0), self.lstmHiddenSize)).to(self.device)  # cell state
        
        temperatures = temperatures.view(temperatures.size(0),temperatures.size(1), temperatures.size(2) * temperatures.size(3))
        output, (h, c) = self.lstm(temperatures, (h, c))

        outputs = [output[:, 0, :], output[:, 10, :], output[:, 20, :], output[:, 30, :]]

        # Concatenate the outputs from all temperature tensors
        lstm_output = torch.stack(outputs, dim=1)  # Shape: (batch_size, 4, hidden_size)
        lstm_output = lstm_output.view(lstm_output.size(0), 1, lstm_output.size(1), -1)

        # Apply convolution to reshape temperature features
        temp_features = self.conv(lstm_output)  # Apply convolution to enhance spatial features
        temp_features = temp_features.view(temp_features.size(0),temp_features.size(2),-1)

        return temp_features


class LSTM(nn.Module):
    def __init__(self, lstmLayersEnc, lstmLayersDec, lstmHiddenSize, lstmInputSize, dropout, attentionHeads, device):
        super(LSTM, self).__init__()
        # global
        self.flatten = nn.Flatten(start_dim = 2, end_dim = 3)

        # LSTM layers encoder
        self.device = device
        self.dropout = dropout
        self.lstmLayersEnc = lstmLayersEnc
        self.lstmLayersDec = lstmLayersDec
        self.lstmHiddenSize = lstmHiddenSize
        self.lstmInputSize = lstmInputSize
        self.lstmEncoder = nn.LSTM(input_size=self.lstmInputSize, hidden_size=self.lstmHiddenSize,
                            num_layers=self.lstmLayersEnc, batch_first=True, dropout = self.dropout)

        # decoder
        self.flattenDec = nn.Flatten(start_dim=1, end_dim=2)
        self.attention = nn.MultiheadAttention(self.lstmInputSize, attentionHeads, self.dropout, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(5000, 2500))

        # Add the Temperature 
        self.temperature_lstm = TemperatureLSTM(self.lstmInputSize,self.lstmHiddenSize, 1, device)

        # Add a query transformation layer
        self.query_transform = nn.Linear(lstmHiddenSize, lstmInputSize)

    def encoder(self, x):
        """
        encodes the input with LSTM cells

        x: torch.tensor
            (b, s, dim)
        return list of torch.tensor and tuple of torch.tensor and torch.tensor
            output, hidden and cell state
        """

        # init hidden and cell state
        h_0 = Variable(torch.zeros(self.lstmLayersEnc, x.size(0), self.lstmHiddenSize)).to(self.device)  # hidden state
        c_0 = Variable(torch.zeros(self.lstmLayersEnc, x.size(0), self.lstmHiddenSize)).to(self.device)  # internal state

        # Propagate input through LSTM
        output, _ = self.lstmEncoder(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        #print('encoder output: ', output.size())
        return output

    def decoder(self, outputEnc, temperatures, y, training):
        """
        applies MHA to output of LSTM encoder

        outputEnc: torch.tensor
        y: torch.tensor
        training: boolean

        returns: torch.tensor

        """
        #print('temperatures lenght: ', len(temperatures))
        #print('temperatures[0].size(): ', temperatures[0].size())
        temp_lstm_output = self.temperature_lstm(temperatures)
        #print('temp_lstm_output.size(): ', temp_lstm_output.size())
        if training == False:
            out = []
            for i in range(4):
                # Transform current state to query
                query = self.query_transform(outputEnc[:, -1:, :])  # Use last timestep as query
                #use the last timestep as query
                # Use encoder output as key and value
                x, _ = self.attention(query, outputEnc, outputEnc)
                x = self.flattenDec(x)
                x = torch.cat((x, temp_lstm_output[:,i,:]), dim=1) 
                x = self.linear(x)
                out.append(x)

                # update hidden input
                outputEnc = torch.cat((outputEnc, x.unsqueeze(dim=1)), dim=1)
                outputEnc = self.encoder(outputEnc[:, -4:, :])

        if training == True:
            y = self.flatten(y)
            out = []
            for i in range(4):
                # Transform current state to query
                query = self.query_transform(outputEnc[:, -1:, :])  # Use last timestep as query
                # Use encoder output as key and value
                x, _ = self.attention(query, outputEnc, outputEnc)
                x = self.flattenDec(x)
                x = torch.cat((x, temp_lstm_output[:,i,:]), dim=1) 
                x = self.linear(x)
                out.append(x)
                
                # update hidden input
                outputEnc = torch.cat((outputEnc, y[:,i,:].unsqueeze(dim =1)), dim=1) #teacher forcing
                outputEnc = self.encoder(outputEnc[:, -4:, :]) #take last 4 timesteps


        out = torch.stack(out, dim = 1)
        out = out.reshape(out.size(dim = 0), out.size(dim = 1), 50, 50)

        

        return out

    def forward(self, x, temperatures, y = None , training = False):

        # flatten input
        x = self.flatten(x)
        s = self.encoder(x)
        output = self.decoder(s, temperatures, y, training)

        # Clip the output to be in the range [0, 1]
        #output = torch.clamp(output, min=0, max=1)

        return output


# test, args: lstmLayersEnc, lstmLayersDec, lstmHiddenSize, lstmInputSize, dropout, attentionHeads, device
device = "cpu"
model = LSTM(1,1, 2500, 2500, 0.1, 1, device).to(device)
Temperatures = torch.rand(8,40, 50,50).to(device)
test = torch.rand(8, 4, 50,50).to(device)

print(model(test,Temperatures, test, training = True).size())







