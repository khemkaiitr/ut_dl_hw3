import torch
import torch.nn.functional as F
import torch.nn as nn


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        loss = F.cross_entropy(input, target)
        return loss


class CNNBlock(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        # Apply batch norm
        self.bn1 = torch.nn.BatchNorm2d(c_out)
        self.conv2 = torch.nn.Conv2d(c_out, c_out, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(c_out)
        # use relu as activation function
        self.relu = torch.nn.ReLU()
        self.use_residual = c_in == c_out

    def forward(self, x):
        x_next = self.relu(self.bn1(self.conv1(x)))
        x_next = self.conv2(x_next)
        x_next = self.bn2(x_next)
        x_next = self.relu(x_next)
        return x_next


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        num_classes = 6
        n_layers = 4
        channel_in = 64
        channel_out = 64

        layers = list()
        layers.append(torch.nn.Conv2d(3, channel_out, 3, padding=1))
        for idx in range(n_layers):
            layers.append(CNNBlock(channel_in, channel_out))
            channel_in = channel_out
            channel_out = channel_out
        self.features = torch.nn.Sequential(*layers)
        self.linear_layers = torch.nn.Linear(channel_in, num_classes)

        # raise NotImplementedError('CNNClassifier.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        x = self.features(x)
        x = x.mean((2, 3))
        out = self.linear_layers(x)
        return out
        # raise NotImplementedError('CNNClassifier.forward')


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        raise NotImplementedError('FCN.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        raise NotImplementedError('FCN.forward')


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
