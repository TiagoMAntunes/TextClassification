import torch
import torch.nn as nn 
initrange = 1

class RNN(nn.Module):
    def __init__(self, embedding_size, n_hidden, hidden_size, output_size, pretrained_embeddings=None, dropout=0):
        super(RNN, self).__init__()

        if pretrained_embeddings is not None:
            self.embeddings = lambda x: torch.tensor(pretrained_embeddings[x]).float()
        else:
            self.embeddings = nn.Embedding(400001, 300)

        self.hidden_states = [torch.zeros(hidden_size)  for _ in range(n_hidden)]
        for state in self.hidden_states:
            state.data.uniform_(-initrange, initrange)

        # concat states and use a bigger matrix in each state for the multiplication
        # (hidden_size + input_size) x hidden_size, hidden_size * 2 x hidden_size, (...)
        self.hidden_layers = nn.ModuleList([nn.Linear(embedding_size + hidden_size, hidden_size)] + [nn.Linear(hidden_size * 2, hidden_size) for _ in range(n_hidden-1)])

        self.hidden_activation = nn.Tanh()        

        self.dropout = nn.Dropout(p=dropout)

        # hidden_size x output_size
        self.reducer = nn.Linear(hidden_size, output_size)

        self.activation = nn.Softmax(dim=-1)

    def forward(self, words, mask):
        """
            Input:
                words   - Tensor Bxd
                mask    - Tensor Bxd

            Returns:
                Tensor B x n x output_size
        """
        
        assert words.shape == mask.shape
        
        # each intermediary state in the network (vertical), repeated to match 
        states = [w.repeat(words.shape[0]).reshape(-1, w.shape[0]) for w in self.hidden_states]
        
        # for each word in the input (horizontal)
        for i in range(words.shape[-1]):
            #  apply each weight
            words_use = self.embeddings(words[:,i])
            mask_use = mask[:,i].unsqueeze(1)
            
            
            # previous state (starts as input embedding)
            prev = words_use
            
            for i,w in enumerate(states):
                join_input = torch.cat((w, prev), dim=-1)
                states[i] = self.dropout(self.hidden_activation(self.hidden_layers[i](join_input))) * mask_use + w * (1 - mask_use)
                prev = states[i]

        return self.activation(self.reducer(states[-1]))