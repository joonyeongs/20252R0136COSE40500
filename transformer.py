# Description: Transformer Model in PyTorch


import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader, dataset
from torch.nn.utils.rnn import pad_sequence

    
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIM = 128
FEED_FORWARD_DIM = 512
NUM_LAYERS = 6
NUM_HEADS = 8
BATCH_SIZE = 512
MAX_LENGTH = 20
LEARNING_RATE = 0.001
EPOCH = 50
SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"
UNK_IDX, SOS_IDX, EOS_IDX, PAD_IDX = 0, 1, 2, 3

    
def tokenize_seq_and_add_special_symbols(raw_text_iter: str, language: str) -> list:
    tokens = tokenizer[language](raw_text_iter)
    tokens = ['<sos>'] + tokens + ['<eos>']

    return tokens

def map_tokens_to_indicies(tokens: list, language: str) -> list:
    indices = [vocab[language][token] for token in tokens]

    return indices
    

def transform_tokens_to_tensor(tokens: list, language: str) -> Tensor:
    indices = map_tokens_to_indicies(tokens, language)
    tensor = torch.tensor(indices, dtype=torch.long)

    return tensor

def longer_than_max_length(tokens: list, language: str) -> bool:
    return len(tokens) > MAX_LENGTH

    
def yield_tokens(tokenized_sentences: list) -> list:
    '''
    Used for building vocabulary
    '''
    for sentence in tokenized_sentences:
        yield sentence


def create_vocab_from_text_data() -> tuple:
    data_for_dict = Multi30k(split="train", language_pair=language_pair) 
    sentences = {SRC_LANGUAGE: [], TGT_LANGUAGE: []}
    vocab, vocab_size = {}, {}

    # Text data is tokenized and added special symbols
    for src_sentence, tgt_sentence in data_for_dict:
        src_sentence = tokenize_seq_and_add_special_symbols(src_sentence, SRC_LANGUAGE)
        tgt_sentence = tokenize_seq_and_add_special_symbols(tgt_sentence, TGT_LANGUAGE)
        if longer_than_max_length(src_sentence, SRC_LANGUAGE) or longer_than_max_length(tgt_sentence, TGT_LANGUAGE):
            continue
        sentences[SRC_LANGUAGE].append(src_sentence)
        sentences[TGT_LANGUAGE].append(tgt_sentence)

    # Build vocabulary
    for lang in language_pair:
        vocab[lang] = build_vocab_from_iterator(yield_tokens(sentences[lang]), specials=special_symbols, min_freq=1)
        vocab[lang].set_default_index(vocab[lang]['<unk>'])
        vocab_size[lang] = len(vocab[lang])

    return vocab, vocab_size
    

def create_dataloader_from_text_data(text_data: dataset , batch_size: int) -> DataLoader:
    src_batch, tgt_batch = [], []

    # Text data is tokenized and padded
    for src_sentence, tgt_sentence in text_data:
        src_sentence = tokenize_seq_and_add_special_symbols(src_sentence, SRC_LANGUAGE)
        tgt_sentence = tokenize_seq_and_add_special_symbols(tgt_sentence, TGT_LANGUAGE)
        if longer_than_max_length(src_sentence, SRC_LANGUAGE) or longer_than_max_length(tgt_sentence, TGT_LANGUAGE):
            continue
        src_sentence = transform_tokens_to_tensor(src_sentence, SRC_LANGUAGE)
        tgt_sentence = transform_tokens_to_tensor(tgt_sentence, TGT_LANGUAGE)

        # Padding
        src_pad_tensor, tgt_pad_tensor = (torch.full((MAX_LENGTH,), PAD_IDX) for _ in range(2))
        src_pad_tensor[:len(src_sentence)] = src_sentence
        tgt_pad_tensor[:len(tgt_sentence)] = tgt_sentence
        src_batch.append(src_pad_tensor)
        tgt_batch.append(tgt_pad_tensor)

    src_batch_tensor = torch.stack(src_batch)
    tgt_batch_tensor = torch.stack(tgt_batch)
    dataset = TensorDataset(src_batch_tensor, tgt_batch_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    return dataloader


def create_padding_mask_for_attention(batch: Tensor, pad_idx: int) -> Tensor:
    padding_mask = (batch != pad_idx)
    sequence_length = batch.size(1)
    padding_mask = padding_mask.unsqueeze(1).repeat(1, sequence_length, 1)
    padding_mask = padding_mask.unsqueeze(1) # Used for multi-head attention
    
    return padding_mask.to(DEVICE) # (batch_size, 1, max_length, max_length)

def create_padding_mask_for_loss(batch: Tensor, pad_idx: int) -> Tensor:
    # Padding mask for loss is used for masking loss of padded tokens
    padding_mask = (batch != pad_idx)

    return padding_mask.to(DEVICE) # (batch_size, max_length)

def create_look_ahead_mask_for_attention(tgt_batch: Tensor) -> Tensor:
    batch_size, tgt_length = tgt_batch.size()
    tgt_attention_mask = torch.tril(torch.ones((tgt_length, tgt_length))).expand(batch_size, tgt_length, tgt_length)
    tgt_attention_mask = tgt_attention_mask.unsqueeze(1) # Used for multi-head attention

    return tgt_attention_mask.to(DEVICE) # (batch_size, 1, max_length, max_length)



def process_batch_for_train(model: nn.Module,
                optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, 
                batch: Tensor,
                ) -> float:
 
    src_batch, tgt_batch = batch
    src_batch, tgt_batch = src_batch.to(DEVICE), tgt_batch.to(DEVICE)
    
    optimizer.zero_grad()
    src_padding_mask = create_padding_mask_for_attention(src_batch, PAD_IDX)
    tgt_padding_mask = create_look_ahead_mask_for_attention(tgt_batch)
    decoder_output = model(src_batch, tgt_batch, src_padding_mask, tgt_padding_mask)

    # Process output and calculate loss
    loss = calculate_loss(decoder_output, tgt_batch, criterion)

    # Backpropagation
    loss.backward()
    optimizer.step()

    return loss.item()

def process_batch_for_test(model: nn.Module,
               criterion: nn.Module, 
               batch: Tensor,
               ) -> float:
   
    src_batch, tgt_batch = batch
    src_batch, tgt_batch = src_batch.to(DEVICE), tgt_batch.to(DEVICE)
    
    src_padding_mask = create_padding_mask_for_attention(src_batch, PAD_IDX)
    tgt_padding_mask = create_look_ahead_mask_for_attention(tgt_batch)
    decoder_output = model.decode(src_batch, src_padding_mask, tgt_padding_mask)

    for i in range(BATCH_SIZE):
        print_translation(src_batch[i], tgt_batch[i], decoder_output[i])


def calculate_loss(decoder_output: Tensor,
                   tgt_batch: Tensor,
                   criterion: nn.Module,
                   ) -> float:
    # Remove <sos> token for target batch
    # decoder_output: (batch_size, max_length, vocab_size)
    # target: (batch_size, max_length)
    decoder_output = decoder_output[:, :-1].reshape(-1, vocab_size[TGT_LANGUAGE])
    target = tgt_batch[:, 1:].reshape(-1)
    tgt_padding_mask = create_padding_mask_for_loss(tgt_batch, PAD_IDX)
    tgt_padding_mask = tgt_padding_mask[:, 1:].reshape(-1)
    loss = criterion(decoder_output, target)
    loss = torch.sum(loss * tgt_padding_mask) / torch.sum(tgt_padding_mask)

    return loss

def train(train_dataloader: DataLoader,
          model: nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: nn.Module,
          ) -> float:
    
    total_loss = 0
    num_batches = len(train_dataloader)
    for batch in train_dataloader:
        loss = process_batch_for_train(model, optimizer, criterion, batch)
        total_loss += loss

    return total_loss / num_batches

def test(test_dataloader: DataLoader,
          model: nn.Module,
          criterion: nn.Module,
          ) -> float:
    
    with torch.no_grad():
        for batch in test_dataloader:
            process_batch_for_test(model, criterion, batch)


def print_translation(src_sequence: Tensor, tgt_sequence: Tensor, output_sequence: Tensor):
    sequences = {'Source': src_sequence, 'Target': tgt_sequence, 'Output': output_sequence}

    for domain, sequence in sequences.items():
        # Truncate before <eos> token
        eos_index = (sequence == EOS_IDX).nonzero(as_tuple=True)
        if len(eos_index) > 0:
            first_eos_index = eos_index[0][0].item()
            sequence_before_eos_tensor = sequence[:first_eos_index]
        else:
            sequence_before_eos_tensor = sequence
            
        if domain == 'Source':   
            # Ex) "Source: A man who just came from a swim"
            print(f"{domain}: {' '.join([vocab[SRC_LANGUAGE].get_itos()[idx] for idx in sequence_before_eos_tensor])}")
        else:
            print(f"{domain}: {' '.join([vocab[TGT_LANGUAGE].get_itos()[idx] for idx in sequence_before_eos_tensor])}")
        
    print("\n")


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, batch):
        embeddings = self.embedding(batch)

        return embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, max_length, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.create_positional_encoding(max_length, embedding_dim)

    def get_angles(self, max_length, embedding_dim):
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        return position / div_term  # (max_length, embedding_dim / 2)

    def create_positional_encoding(self, max_length, embedding_dim):
        angles = self.get_angles(max_length, embedding_dim)
        positional_encoding = torch.zeros(max_length, embedding_dim)
        positional_encoding[:, 0::2] = torch.sin(angles)
        positional_encoding[:, 1::2] = torch.cos(angles)

        return positional_encoding

    def forward(self, x):
        self.positional_encoding = self.positional_encoding.to(x.device)
        return x + self.positional_encoding[:x.size(1), :].detach()

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.scale = torch.sqrt(torch.tensor(attention_dim).float())

    def forward(self, query, key, value, mask=None):
        key = key.transpose(-2, -1)
        attention_score = torch.matmul(query, key) / self.scale # (batch_size, num_heads, max_length, max_length)
        attention_score = attention_score.masked_fill(mask == 0, -1e10)
        attention_distribution = torch.softmax(attention_score, dim=-1)
        attention_value = torch.matmul(attention_distribution, value) # (batch_size, num_heads, max_length, attention_dim)

        return attention_value

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self,).__init__()
        self.attention_dim = embedding_dim // num_heads
        assert self.attention_dim * num_heads == embedding_dim

        self.scaled_dot_product_attention = ScaledDotProductAttention(embedding_dim, self.attention_dim)
        self.Wq = nn.Linear(embedding_dim, self.attention_dim * num_heads)
        self.Wk = nn.Linear(embedding_dim, self.attention_dim * num_heads)
        self.Wv = nn.Linear(embedding_dim, self.attention_dim * num_heads)
        self.Wo = nn.Linear(embedding_dim, embedding_dim)
        self.num_heads = num_heads

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        max_length = query.size(1)

        query = self.Wq(query).reshape(batch_size, max_length, self.num_heads, self.attention_dim).transpose(1, 2)
        key = self.Wk(key).reshape(batch_size, max_length, self.num_heads, self.attention_dim).transpose(1, 2)
        value = self.Wv(value).reshape(batch_size, max_length, self.num_heads, self.attention_dim).transpose(1, 2)

        attention_values = self.scaled_dot_product_attention(query, key, value, mask)   # (batch_size, num_heads, max_length, attention_dim)
        attention_values = attention_values.transpose(1, 2).reshape(batch_size, max_length, self.num_heads * self.attention_dim)
        output = self.Wo(attention_values)

        return output
        

class LayerNormalization(nn.Module):
    def __init__(self, num_features):
        super(LayerNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x):
        x = self.layer_norm(x)
        return x

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, embedding_dim, feed_forward_dim):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.W1 = nn.Linear(embedding_dim, feed_forward_dim)
        self.W2 = nn.Linear(feed_forward_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        outputs = self.W1(inputs)
        outputs = self.relu(outputs)
        outputs = self.W2(outputs)

        return outputs

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, feed_forward_dim, num_heads):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.feed_forward_neural_network = FeedForwardNeuralNetwork(embedding_dim, feed_forward_dim)
        self.norm1 = LayerNormalization(embedding_dim)
        self.norm2 = LayerNormalization(embedding_dim)

    def forward(self, batch, padding_mask):
        layer_1 = batch + self.multi_head_attention(batch, batch, batch, padding_mask)
        normalized_layer_1 = self.norm1(layer_1)

        layer_2 = normalized_layer_1 + self.feed_forward_neural_network(normalized_layer_1)
        output = self.norm2(layer_2)

        return output
    
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, feed_forward_dim, num_heads):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.feed_forward_neural_network = FeedForwardNeuralNetwork(embedding_dim, feed_forward_dim)
        self.norm1 = LayerNormalization(embedding_dim)
        self.norm2 = LayerNormalization(embedding_dim)
        self.norm3 = LayerNormalization(embedding_dim)

    def forward(self, batch, encoder_output, padding_mask, attention_mask):
        layer_1 = batch + self.masked_multi_head_attention(batch, batch, batch, attention_mask)
        normalized_layer_1 = self.norm1(layer_1)

        layer_2 = normalized_layer_1 + self.multi_head_attention(normalized_layer_1, encoder_output, encoder_output, padding_mask)
        normalized_layer_2 = self.norm2(layer_2)

        layer_3 = normalized_layer_2 + self.feed_forward_neural_network(normalized_layer_2)
        output = self.norm3(layer_3)

        return output

class StackedEncoder(nn.Module):
    def __init__(self, vocab_size, max_length, embedding_dim, feed_forward_dim, num_heads, num_layers):
        super(StackedEncoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(max_length, embedding_dim)
        self.encoder_layers = nn.ModuleList([EncoderLayer(embedding_dim, feed_forward_dim, num_heads)
                                             for _ in range(num_layers)])

    def forward(self, batch, padding_mask):
        embedding = self.embedding(batch)
        output = self.positional_encoding(embedding)

        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output, padding_mask)

        return output

class StackedDecoder(nn.Module):
    def __init__(self, vocab_size, max_length, embedding_dim, feed_forward_dim, num_heads, num_layers):
        super(StackedDecoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(max_length, embedding_dim)
        self.decoder_layers = nn.ModuleList([DecoderLayer(embedding_dim, feed_forward_dim, num_heads)
                                             for _ in range(num_layers)])
        self.dense_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, batch, encoder_output, padding_mask, attention_mask):
        embedding = self.embedding(batch)
        output = self.positional_encoding(embedding)

        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, encoder_output, padding_mask, attention_mask)

        output = self.dense_layer(output)

        return output
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, max_length, embedding_dim, feed_forward_dim, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.encoder = StackedEncoder(vocab_size[SRC_LANGUAGE], max_length, embedding_dim, feed_forward_dim, num_heads, num_layers)
        self.decoder = StackedDecoder(vocab_size[TGT_LANGUAGE], max_length, embedding_dim, feed_forward_dim, num_heads, num_layers)

    def decode(self, src_batch, src_padding_mask, tgt_attention_mask):
        batch_size = src_batch.size(0)
        max_length = src_batch.size(1)

        decoder_input = torch.zeros(batch_size, max_length, dtype=torch.long).to(DEVICE)
        decoder_input[:, 0] = SOS_IDX
        decoder_output = torch.zeros(batch_size, max_length, dtype=torch.long).to(DEVICE)

        tgt_attention_mask_step = torch.zeros_like(tgt_attention_mask)

        encoder_output = self.encoder(src_batch, src_padding_mask)
        for step in range(max_length):
            # Masked attention mask so that decoder can only see predicted tokens
            tgt_attention_mask_step[:, :, :step+1, :step+1] = tgt_attention_mask[:, :, :step+1, :step+1]

            decoder_output_tensor = self.decoder(decoder_input, encoder_output, src_padding_mask, tgt_attention_mask_step)
            decoder_output_tensor = decoder_output_tensor[:, step, :]
            decoder_output[:, step] = decoder_output_tensor.argmax(dim=-1)

            if step < (max_length - 1):
                decoder_input[:, step+1] = decoder_output[:, step]

        return decoder_output

    def forward(self, src_batch, tgt_batch, src_padding_mask, tgt_attention_mask):
        encoder_output = self.encoder(src_batch, src_padding_mask)
        decoder_output = self.decoder(tgt_batch, encoder_output, src_padding_mask, tgt_attention_mask)

        return decoder_output
 

if __name__ == "__main__":
    language_pair = (SRC_LANGUAGE, TGT_LANGUAGE)
    special_symbols = ['<unk>', '<sos>', '<eos>', '<pad>']
    tokenizer = {SRC_LANGUAGE : get_tokenizer('spacy', language='de_core_news_sm'),
                 TGT_LANGUAGE : get_tokenizer('spacy', language='en_core_web_sm')}

    # Each is a dictionary which has key: language
    vocab, vocab_size = create_vocab_from_text_data()
    
    # Multi30k test set has encoding problem, so we use train and valid set for training and testing
    train_data, test_data = Multi30k(split=('train', 'valid'), language_pair=language_pair)        
    train_dataloader = create_dataloader_from_text_data(train_data, BATCH_SIZE)
    test_dataloader = create_dataloader_from_text_data(test_data, BATCH_SIZE)
    
    
    # Training
    transformer = Transformer(vocab_size, MAX_LENGTH, MODEL_DIM, FEED_FORWARD_DIM, NUM_HEADS, NUM_LAYERS).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction='none')
    transformer_optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCH):
        train_loss = train(train_dataloader, transformer, transformer_optimizer, criterion)
        
        if (epoch + 1) % (EPOCH / 10) == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(train_loss))
    
    # Test
    test(test_dataloader, transformer, criterion)
    

        
    
    