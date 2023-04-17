import torch
import torch.nn as nn
import math

# define the Multiple Attention Class Layer

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super().__init__()

        self.num_heads = num_heads

        self.query_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.keys_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.values_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)

        self.output_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)

    def forward(self, query, keys, values):

        batch_size, sequence_length, embedding_dim = query.shape

        head_dim = embedding_dim // self.num_heads

        # compute the query projection
        # shape: [batch_size, sequence_length, embedding_dim]
        query = self.query_proj(query)
        # split the query into multiple heads
        query = query.reshape(batch_size, sequence_length, self.num_heads, head_dim)
        # permute the query to shape: [batch_size, num_heads, sequence_length, head_dim]
        query = query.permute(0, 2, 1, 3)
        # reshape the query to shape: [batch_size * num_heads, sequence_length, head_dim]
        query = query.reshape(batch_size * self.num_heads, sequence_length, head_dim)

        keys = self.keys_proj(keys)
        # split the keys into multiple heads
        keys = keys.reshape(batch_size, sequence_length, self.num_heads, head_dim)
        # permute the keys to shape: [batch_size, num_heads, sequence_length, head_dim]
        keys = keys.permute(0, 2, 1, 3)
        # reshape the keys to shape: [batch_size * num_heads, sequence_length, head_dim]
        keys = keys.reshape(batch_size * self.num_heads, sequence_length, head_dim)

        values = self.values_proj(values)
        # split the values into multiple heads
        values = values.reshape(batch_size, sequence_length, self.num_heads, head_dim)
        # permute the values to shape: [batch_size, num_heads, sequence_length, head_dim]
        values = values.permute(0, 2, 1, 3)
        # reshape the values to shape: [batch_size * num_heads, sequence_length, head_dim]
        values = values.reshape(batch_size * self.num_heads, sequence_length, head_dim)

        # compute the attention scores, this gives attention scores of [batch_size, sequence_length, sequence_length]
        attention_scores = torch.bmm(query, keys.transpose(1, 2))

        # normalize the attention scores by the square root of the embedding dimension
        attention_scores = attention_scores / math.sqrt(embedding_dim)

        # compute softmax over the word word importance/attention map

        norm_attention_scores = torch.softmax(attention_scores, dim=-1)

        # compute the weighted sum of values
        # this gives us the output of shape [batch_size, sequence_length, embedding_dim]
        output = torch.bmm(norm_attention_scores, values)

        # reshape the output to shape: [batch_size, num_heads, sequence_length, head_dim]
        output = output.reshape(batch_size, self.num_heads, sequence_length, head_dim)
        # permute the output to shape: [batch_size, sequence_length, num_heads, head_dim]
        output = output.permute(0, 2, 1, 3)
        # reshape the output to shape: [batch_size, sequence_length, embedding_dim]
        output = output.reshape(batch_size, sequence_length, self.num_heads * head_dim)

        # because we compute them as separate heads, use a final linear layer to combine them
        output = self.output_proj(output)

        return output


# define the class Transformer Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads = 4):
        super().__init__()
        self.attention = MultiHeadAttentionLayer(embedding_dim = embedding_dim, num_heads=num_heads)

        # normalixe Attention layer with Layer Norm
        self.attn_norm = nn.LayerNorm(embedding_dim)

        # build the layers
        self.ffn_layers = nn.Sequential(
            nn.Linear(in_features = embedding_dim, out_features = embedding_dim * 4),
            nn.GELU(), 
            nn.Linear(in_features = embedding_dim *4 , out_features = embedding_dim)
        )

        self.ffn_norm = nn.LayerNorm(embedding_dim)


    def forward(self, x):
        # compute the layer norm of the input
        x = self.attn_norm(x)

        # compute output by performing a residual connection
        # with norm input and the result from the attention norm input passed
        # into the attention layer

        x = x + self.attention(x, x, x)

        # compute the normalization of the attention output
        x = self.ffn_norm(x)

        # compute the final ouput by performing a residual connection
        # with norm attention output and the result from the norm output passed
        # into the ffn layer
        output = x + self.ffn_layers(x)

        return output


# define class for Classifier Model
class TransformerClassifierModel(nn.Module):
    def __init__(self, vocab_size, max_len, embedding_dim = 128,  depth = 6, num_classes = 4):
        super().__init__()
        
        # define the tokenizer layer
        self.tokenizer = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)
        # define  the Positionally encoding layer
        self.postion_encoding = nn.Embedding(num_embeddings = max_len, embedding_dim = embedding_dim)

        # define the attention layer
        self.attention_layers = nn.ModuleList(TransformerEncoderBlock(embedding_dim = embedding_dim) for _ in range(depth))

        # define the classifier layer
        self.classifier = nn.Linear(in_features = embedding_dim, out_features = num_classes)
        
        # register word positions for word encoding
        self.register_buffer("positions", torch.arange(max_len))


    def forward(self, input):
        """
        Steps:
        Tokenize words and embed each word with an embedding dimension
        Positional encoding of words with same embedding dim
        Pass the feature vectors of these words into the Attention Layer
        """
        # tokenize the words
        # shape changes from [batch_size, max_len] -> [batch_size, max_len, embedding dim]
        # e.g 
        tokenized_words = self.tokenizer(input)

        # get the positions of each of the tokenized words
        positions_of_input = self.positions[:input.shape[1]]

        # get the positional embedding of each word
        position_embeddings = self.postion_encoding(positions_of_input)

        # concatenate the positonal embeddings with the feature vectors from the tokenized words

        output = tokenized_words + position_embeddings

        # pass the input into the attention layer
        for layer in self.attention_layers:
            output = layer(output)


        #compute the mean of all the words
        output = torch.mean(output, dim = 1)

        # obtain the class prediction
        output = self.classifier(output)

        return output











        


