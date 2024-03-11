import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def forward(self, prompt, response):
        hidden_state = self.encoder(prompt)
        vocab_size = self.decoder.input_size
        outputs = torch.zeros(len(response), vocab_size)
        response_inp = response[:-1]
        for i in range(1, len(response_inp)):
            # teacher forcing
            inp = torch.tensor(response_inp[i-1]).unsqueeze(0).to(self.device)
            out, hidden_state = self.decoder(inp, hidden_state)
            outputs[i] = out.squeeze(0)
        
        return outputs

class Encoder(nn.Module):
    def __init__(self, vocab_size, config):
        """
        Initialize the encoder, consisting of an embedding layer and LSTM layer
        """
        super(Encoder, self).__init__()
        self.input_size = vocab_size
        self.hidden_size = config['hidden_size']
        self.embedding_dim = config['embedding_dim']
        self.n_layers = config['n_layers']
        self.dropout_p = config['dropout']
        self.hidden = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Embedding layer
        self.embedding = nn.Embedding(self.input_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        # LSTM layer
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.n_layers)
        
    def init_hidden(self):
        """
        Initializes the hidden state for the recurrent neural network.
        """
        # initialize hidden state
        h0 = torch.zeros(self.n_layers, self.hidden_size).to(self.device) # for unbatched input 
            
        # initialize cell state
        c0 = torch.zeros(self.n_layers, self.hidden_size).to(self.device)
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
        x1 = self.dropout(x0)
        _, self.hidden = self.rnn(x1, self.hidden)

        return self.hidden
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, config):
        """
        Initialize the Decoder model.
        """
        super(Decoder, self).__init__()
        self.input_size = vocab_size
        self.embedding_dim = config['embedding_dim']
        self.hidden_size = config['hidden_size']
        self.n_layers = config['n_layers']
        self.dropout_p = config['dropout']
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Embedding layer
        self.embedding = nn.Embedding(self.input_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=self.dropout_p)
        # LSTM layer
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.n_layers)
        # Dense linear layer
        self.linear = nn.Linear(self.hidden_size, self.input_size) # input and output should be the same size
        
    def forward(self, seq, hidden_states):
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
        x1 = self.dropout(x0)
        x2, hidden_states = self.rnn(x1, hidden_states)
        out = self.linear(x2)

        return out, hidden_states