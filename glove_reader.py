import numpy as np

class GloveReader:
    def __init__(self, path='glove.6B.300d.txt'):
        words = []
        embeddings = []
        with open(path) as f:
            for line in f:
                line = line.split()
                words.append(line[0])
                embeddings.append(list(map(float, line[1:])))
            
        self.embeddings = np.array(embeddings)
        del embeddings
        
        self.words2idx = {j:i for i,j in enumerate(words)}
        self.idx2words = {i:j for i,j in enumerate(words)}
        del words
        