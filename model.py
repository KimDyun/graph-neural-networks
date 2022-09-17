import torch.nn as nn
from torch_geometric.nn import ChebConv
from torch_geometric.transforms import LaplacianLambdaMax


class DGCNN(nn.Module):
    def __init__(self, num_features, hid_channels, out_channels, k,  edge_weight, batch_size, learnable=False):
        super(DGCNN, self).__init__()

        self.lambdamax = LaplacianLambdaMax(None)

        self.in_channels = num_features
        self.cheb_out_channels = hid_channels
        self.conv1d_out_channels = int(hid_channels/2)
        self.fc1_out_channels = int(self.conv1d_out_channels/2)
        self.out_channels = out_channels

        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learnable)
        self.batch_size = batch_size

        self.chebconv1 = ChebConv(self.in_channels, self.cheb_out_channels, K=k, normalization=None)

        self.conv1d = nn.Conv1d(self.cheb_out_channels, self.conv1d_out_channels, kernel_size=1)

        self.fc1 = nn.Linear(self.conv1d_out_channels, self.fc1_out_channels)
        self.fc2 = nn.Linear(self.fc1_out_channels, self.out_channels)
        self.fc_module = nn.Sequential(self.fc1, self.fc2)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, data, _type=None):
        data.edge_attr = self.edge_weight.data.repeat(self.batch_size)
        data = self.lambdamax(data)

        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(dim=1)

        cheb_layer = self.cheb_conv1(data.x, data.edge_index, self.relu(data.edge_attr), lambda_max=data.lambda_max)

        conv1d_layer = self.relu(self.conv1d(cheb_layer))
        flatten = conv1d_layer.reshape(self.batch_size, -1)

        fc_layer = self.fc_module(flatten)

        if _type == 'train':
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(fc_layer.view(-1, self.num_classes), data.y.view(-1))
            return loss
        else:
            logits = self.softmax(fc_layer)
            return logits

