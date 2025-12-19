# Description: Neural Network Language Model in PyTorch

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader, dataset
from datetime import datetime


def create_vocab_from_text_data() -> tuple:
    data_for_dict = WikiText2(split='train')    
    vocab = build_vocab_from_iterator(map(tokenizer, data_for_dict), specials=['<unk>'], min_freq=3)
    vocab.set_default_index(vocab['<unk>'])
    vocab_size = len(vocab)

    return vocab, vocab_size
    

def tokenize_and_split(raw_text_iter: dataset.IterableDataset, length_of_sequence: int) -> Tensor:
    """
    1. Tokenize and convert raw text into a list of tensors, where each tensor represents a sequence of tokens
    2. Eliminate empty tensors and concatenate into a single tensor
    3. Split the single tensor into samples, where each sample represents a sequence of tokens
    """
    
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    total_samples = data.shape[0] // length_of_sequence 
    data = data[:length_of_sequence * total_samples]                       
    data = data.view(total_samples, length_of_sequence)   
           
    return data

def create_dataloader_from_text_data(text_data_tensor: Tensor, batch_size: int) -> DataLoader:
    dataset = TensorDataset(text_data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader

def split_batch_into_input_and_target(batch: list) -> tuple:
    input_batch, target_batch = batch[0][:, :-1], batch[0][:, -1]
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    return input_batch, target_batch


def train():
    train_loss = 0
    for batch in train_dataloader:
        input_batch, target_batch = split_batch_into_input_and_target(batch)

        optimizer.zero_grad()
        output = model(input_batch)
        train_loss = criterion(output, target_batch)
        train_loss.backward()
        optimizer.step()
    
    return train_loss


def test():
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    for batch in test_dataloader:
        input_batch, target_batch = split_batch_into_input_and_target(batch)

        output = model(input_batch)
        loss = criterion(output, target_batch)
        test_loss += loss.item()
        correct += (output.argmax(1) == target_batch).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return test_loss, correct

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        """
        y = b + Wx + U * tanh(d + Hx)
        """
        self.n_steps = length_of_sequence - 1
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.h = nn.Linear(self.n_steps * embedding_dim, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.tanh = nn.Tanh()
        self.u = nn.Linear(n_hidden, vocab_size, bias=False)
        self.w = nn.Linear(self.n_steps * embedding_dim, vocab_size, bias=False)
        self.b = nn.Parameter(torch.ones(vocab_size))
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, self.n_steps * embedding_dim)     # need to check this
        tanh = self.tanh(self.d + self.h(x))
        y = self.b + self.w(x) + self.u(tanh)
        return y
    

if __name__ == "__main__":
    start_time = datetime.now()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_dim = 100
    length_of_sequence = 10
    n_hidden = 60
    batch_size = 64
    tokenizer = get_tokenizer('basic_english')

    vocab, vocab_size = create_vocab_from_text_data()
    train_iter, test_iter = WikiText2(split='train'), WikiText2(split='test')
    train_data = tokenize_and_split(train_iter, length_of_sequence)
    test_data = tokenize_and_split(test_iter, length_of_sequence)
    train_dataloader = create_dataloader_from_text_data(train_data, batch_size)
    test_dataloader = create_dataloader_from_text_data(test_data, batch_size)
   
    
    # Training
    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        train_loss = train()

        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(train_loss))

    # Test
    with torch.no_grad():
        test_loss, accuracy = test()

    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    end_time = datetime.now()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time}")

    

        
    
    
