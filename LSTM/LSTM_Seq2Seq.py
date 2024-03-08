import torch
import torch.nn as nn
from constants import *


class Encoder(nn.Module):
    def __init__(self, input_size, config):
        """
        Initialize the encoder, consisting of an embedding layer and LSTM layer
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = config['hidden_size']
        self.n_layers = config['n_layers']
        self.hidden = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Embedding layer
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        
        # LSTM layer
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers)
        
    def init_hidden(self):
        """
        Initializes the hidden state for the recurrent neural network.
        """
        # initialize hidden state
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(self.device) # for unbatched input 
            
        # initialize cell state
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(self.device)
        self.hidden = (h0, c0)

        
    def forward(self, seq):
        """
        Forward pass of the encoder.
        
        Parameters:
        - seq (Tensor): Input sequence tensor of shape (seq_length)

        Returns:
        - encoder_hidden: Hidden states of the encoder to be passed into decoder

        (i) Embed the input sequence
        (ii) Forward pass through the recurrent layer
        """
        x0 = self.embedding(seq)
        _, self.hidden = self.rnn(x0, self.hidden)

        return self.hidden
    
class Decoder(nn.Module):
    def __init__(self, input_size, config):
        """
        Initialize the Decoder model.
        """
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = config['hidden_size']
        self.n_layers = config['n_layers']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Embedding layer
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        # LSTM layer
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers)
        # Dense linear layer
        self.linear = nn.Linear(self.hidden_size, self.input_size) # input and output should be the same size
        
    def forward(self, seq, encoder_hidden_states):
        """
        Forward pass of the decoder

        Parameters:
        - seq (Tensor): Input sequence tensor of shape (seq_length)
        - encoder_hidden_states: Hidden states from encoder

        Returns:
        - output (Tensor): Output tensor of shape (input_size)

        (i) Embed the input sequence
        (ii) Forward pass through the recurrent layer
        (iv) Pass through the linear output layer
        """
        x0 = self.embedding(seq)
        x1, _ = self.rnn(x0, encoder_hidden_states)
        out = self.linear(x1)

        return out