import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, kernel_sizes, num_channels, embed_size, num_class, dropout=0.2):
        super(TextCNN,self).__init__()
        self.embed_size = embed_size
        self.kernel_sizes = kernel_sizes
        self.num_channels = num_channels
        self.num_class = num_class
        self.encoder_layers = nn.ModuleList()
        for kernel,channel in zip(self.kernel_sizes,self.num_channels):
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=self.embed_size,out_channels=channel,kernel_size=kernel),
                    nn.BatchNorm1d(num_features=channel),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveMaxPool1d(1)
                )
            )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=sum(self.num_channels),out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256,out_features=num_class),
            nn.Dropout(p=dropout)
        )
    
    def forward(self,inputs):
        """
        inputs 的形状为(B,seq_length,embed_size)
        """
        assert inputs.shape[2] == self.embed_size
        inputs = inputs.permute(0,2,1)
        out = torch.cat([layer(inputs).squeeze(2) for layer in self.encoder_layers],dim=1)
        return self.fc_layer(out)


class BiGru(nn.Module):
    def __init__(self,hidden_size,num_class,embed_size,num_layers):
        super(BiGru,self).__init__()
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers

        self.encoder = nn.GRU(input_size=self.embed_size,hidden_size=self.hidden_size,bidirectional=True,batch_first=True,num_layers=self.num_layers)

        self.fc_layer = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size),nn.ReLU(inplace=True),nn.Linear(self.hidden_size,self.num_class))

    def forward(self,inputs):
        assert inputs.shape[2] == self.embed_size
        out,hn = self.encoder(inputs)
        return self.fc_layer(out[:,-1,:])

