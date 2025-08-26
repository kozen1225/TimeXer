
import torch
import torch.nn as nn

class TrendExtractor(nn.Module):
    """
    Decomposes the input series into trend and residual components using a moving average.
    """
    def __init__(self, kernel_size=25):
        super(TrendExtractor, self).__init__()
        self.kernel_size = kernel_size
        # Use AvgPool1d as a simple moving average. Padding is set to handle boundaries.
        self.moving_avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=int((self.kernel_size - 1) / 2))

    def forward(self, x):
        """
        Forward pass for TrendExtractor.
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Channels, Length]
                              In the context of TimeXer, this will be [B*n_vars, D, N]
        Returns:
            torch.Tensor: The trend component.
            torch.Tensor: The residual component.
        """
        # Calculate the trend component by applying the moving average
        trend = self.moving_avg(x)
        # Calculate the residual component by subtracting the trend from the original series
        residual = x - trend
        return trend, residual
