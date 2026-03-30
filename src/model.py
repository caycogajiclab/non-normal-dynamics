
import torch
import torch.nn as nn
import numpy as np

torch.set_default_dtype(torch.float64) # for higher precision


class SequentialRNN(nn.Module):
    """

    Apply a single layer RNN to an input sequence. The RNN is defined as:

    .. math::
        h_{t+1} = h_t + \tau\phi(h_t W_{hh}^T + W_{ih}x_t + \varepsilon)
    
    where :math:`\tau` is the time step, :math:`\phi` is the activation function, :math:`W_{hh}` are the recurrent weights, :math:`W_{ih}` are the input weights and :math:`\varepsilon` is the extrinsic noise. 
    The input weights are initialized to pass the input signal to the first mode of the recurrent dynamics
    
    can be optionally rotated by a random orthogonal matrix. 
    The model can also include intrinsic noise by adding noise to the recurrent weights.


    Args:
        input_dim: number of expected dimensions in input x
        hidden_dim: number of expected dimensions in hidden state
        activation_function: activation function to use. Defaults: ``linear``
        rotation: whether to rotate the hidden output. Default: ``False``
        extrinsic_noise: standard deviation of noise added to the hidden output. Default: 0.
        intrinsic_noise: standard deviation of noise added to the recurrent weights. Default: 0.
        amplification: amplification factor for input weights. Default: 10.
    """
    def __init__(self, input_dim, hidden_dim, activation_function='linear', rotation=False, extrinsic_noise=0., intrinsic_noise=0., amplification=10):
        super().__init__()
        self.W_hh = nn.Linear(hidden_dim,hidden_dim,bias=False)
        torch.nn.init.uniform_(self.W_hh.weight, a=0.0001, b=0.1) # weight initialization
        self.h0 = nn.Parameter(torch.zeros(hidden_dim), requires_grad=False) # fixed initial hidden state
        self.dt = nn.Parameter(torch.tensor(0.1)) # learnable time step
        self.extrinsic_noise = extrinsic_noise 
        self.intrinsic_noise = intrinsic_noise 
        self.W_ih = torch.tensor(np.identity(hidden_dim)*amplification)[:input_dim] # first Schur Mode, input signal has to be amplified
        self.rotation = rotation
        if self.rotation:
            Q, _ = np.linalg.qr(np.random.randn(hidden_dim, hidden_dim))
            self.D = torch.tensor(Q, requires_grad=False, dtype=torch.float64)
            self.register_buffer('Q', self.D)
            self.W_ih = self.W_ih @ self.D[0] 
        match activation_function:
          case 'linear':
            self.activation_function=nn.Identity()
          case 'relu':
            self.activation_function=nn.ReLU()
          case 'tanh':
            self.activation_function=torch.tanh
          case _:
            raise ValueError(f"Unknown activation function '{activation_function}'. Select from 'linear', 'relu' or 'tanh'.")

    def forward(self, x, W_hh=None):
        batch_size, seq_len, _ = x.shape
        h0 = self.h0.unsqueeze(0).repeat(batch_size, 1)  # (batch, hidden_dim)
        W_ih = self.W_ih.unsqueeze(0).repeat(batch_size, 1)
        W_hh = W_hh or self.W_hh.weight
        time_step = 0.001 + (0.1 - 0.001) * torch.nn.Sigmoid()(self.dt)
        outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]  # extract time slice for batch_first (batch, input_dim)
            h = h0 + self.activation_function(time_step*(h0 @ W_hh.T + W_ih * xt))
            h0 = h
            y = h
            if self.rotation:
                y = y  @ self.D + self.extrinsic_noise * torch.randn_like(y) # add extrinsic noise in the rotated space
            else:
                y = y + self.extrinsic_noise * torch.randn_like(y) # add extrinsic noise in the original space
            outputs.append(y.unsqueeze(1)) 

        return torch.cat(outputs, dim=1)  # (batch, seq, hidden_dim)


