import torch.nn as nn



class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """DQN Network
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        """Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
        """
        x = self.fc1(x)
        x = self.fc2(x)
        y = self.fc3(x)
        return y
