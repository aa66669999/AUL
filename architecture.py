import math

import timm
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from edl_pytorch import NormalInvGamma
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import VisionTransformer
from transformers.models import vit


class ResNet18(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, embed,
                 drop_p, activation):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        # Freeze the parameters of the ResNet-18 backbone
        for param in self.resnet.parameters():
            param.requires_grad = True

        # Resize input to match the expected input size of ResNet-18
        # self.fc_input = nn.Linear(in_size, 224)  # Adjust input size to match ResNet-18

        # Modify the classifier layers to match the desired output size
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        # # Resize input to match the expected input size of ResNet-18
        # x = self.fc_input(x)

        # Forward pass through the ResNet-18 backbone
        #x.cuda()
        results = self.resnet(x)
        return results


class BernoulliResnet18(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, embed_length, drop_p=0.5, activation=nn.ELU()):
        super(BernoulliResnet18, self).__init__()

        self.k = 6
        self.embed_length = embed_length
        # ResNet18 embedding
        self.resnet_embedding = ResNet18(in_size, hidden_size, self.embed_length*self.k,
                                         embed_length, drop_p, activation)

        self.out_size = out_size
        # Embedding layer

        # self.embed = nn.Embedding(embed_length, self.k)
        # self.embed = nn.Embedding(embed_length, self.k)

        # Fully connected layers for Pi
        self.fc_pi = nn.Sequential(
            nn.Linear(self.embed_length*self.k, hidden_size),
            activation,
            nn.Dropout(drop_p),
            nn.Linear(hidden_size, self.k)
        )

        # Fully connected layers for Miu
        self.fc_miu = nn.Sequential(
            nn.Linear(embed_length, hidden_size),
            activation,
            nn.Dropout(drop_p),
            nn.Linear(hidden_size, out_size),  # Adjusted for k*out_size
            nn.Sigmoid()  # Sigmoid activation for Miu
        )

    def forward(self, x):
        print("Shape of our data batch shape:", x.shape)
        # Obtain ResNet-18 embeddings
        resnet_embeddings = self.resnet_embedding(x)
        embedding_miu = resnet_embeddings.view(-1, self.embed_length, self.k)
        print("Shape before embedding layer:", embedding_miu.shape)

        # Component means (Miu)
        # Convert input tensor to Long type
        print("Shape before embedding layer:", embedding_miu.shape)
        # Pass through the embedding layer
        # embedding_miu = self.embed(embedding_miu)
        embedding_miu = embedding_miu.permute(0, 2, 1)
        print("Shape after embedding layer:", embedding_miu.shape)

        # Concatenate ResNet-18 embeddings with previous embeddings
        # embedding_miu = torch.cat((embedding_miu, resnet_embeddings.unsqueeze(1)), dim=1)

        miu = self.fc_miu(embedding_miu)

        # Component probabilities (Pi)
        pi = torch.softmax(self.fc_pi(resnet_embeddings), dim=-1)

        pi_expanded = pi.unsqueeze(-1).expand(-1, -1, self.out_size)

        # Multiply miu and pi_expanded
        result = torch.sum(miu * pi_expanded, dim=1)

        return result


class BernoulliMixture(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, embed_length, drop_p=0.5, activation=nn.ELU()):
        super(BernoulliMixture, self).__init__()

        self.k = 10
        self.embed = nn.Embedding(in_size, self.k)

        # Fully connected layers for Pi
        self.fc_pi = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            activation,
            nn.Dropout(drop_p),
            nn.Linear(hidden_size, self.k)
        )

        # Fully connected layers for Miu
        self.fc_miu = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            activation,
            nn.Dropout(drop_p),
            nn.Linear(hidden_size, out_size),  # Adjusted for k*out_size
            nn.Sigmoid()  # Sigmoid activation for Miu
        )

    def forward(self, x):
        # Component means (Miu)
        # Convert input tensor to Long type
        embedding_miu = x.long()
        print("Shape before embedding layer:", embedding_miu.shape)
        # Pass through the embedding layer
        embedding_miu = self.embed(embedding_miu)
        embedding_miu = embedding_miu.permute(0, 2, 1)
        print("Shape after embedding layer:", embedding_miu.shape)

        miu = self.fc_miu(embedding_miu)
        # print("Shape of miu", miu.shape)

        # Component probabilities (Pi)
        pi = torch.softmax(self.fc_pi(x), dim=-1)
        # print("Shape of pi:", pi.shape)

        pi_expanded = pi.unsqueeze(-1).expand(-1, -1, 156)

        # Multiply miu and pi_expanded
        result = torch.sum(miu * pi_expanded, dim=1)  # Sum along the second dimension of miu
        # print("Shape of result after reshaping:", result.shape)

        # # Compute the final result: sum_k^6{Pi_k * Miu_k}
        # result = torch.sum(pi.unsqueeze(-1) * miu, dim=1)

        return result


class ViTModel(nn.Module):
    def __init__(self, in_size=None, hidden_size=None, out_size=None, embed=None,
                 drop_p=0.5, activation=None):
        super(ViTModel, self).__init__()

        # Load the pretrained Vision Transformer model
        self.vit = VisionTransformer(img_size=224, patch_size=16, num_classes=512, embed_dim=embed,
                                     depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True, drop_rate=drop_p)

        # Modify the classification head to match the desired output size
        self.fc = nn.Sequential(
            nn.Linear(self.vit.embed_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        # Forward pass through the Vision Transformer backbone
        features = self.vit(x)
        # Apply the classification head
        output = self.fc(features)
        return output

#
# class ViTModel(nn.Module):
#     def __init__(self, in_size, hidden_size, out_size, embed,
#                  drop_p, activation):
#         super(ViTModel, self).__init__()
#
#         # Load the pretrained Vision Transformer model
#         pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
#         self.vit = pretrained_vit
#
#         # Freeze the base parameters
#         for parameter in self.vit.parameters():
#             parameter.requires_grad = True
#
#         # Modify the final fully connected layer
#         self.vit.head = nn.Linear(in_features=self.vit.head.in_features, out_features=156, bias=True)
#
#     def forward(self, x):
#         # Forward pass through the Vision Transformer backbone
#         outputs = self.vit(x)
#
#         return outputs


class ResNet50(nn.Module):
    def __init__(self, in_size=None, hidden_size=None, out_size=None, embed=None,
                 drop_p=0.5, activation=None):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Freeze the parameters of the ResNet-18 backbone
        for param in self.resnet.parameters():
            param.requires_grad = True

        # Resize input to match the expected input size of ResNet-18
        self.fc_input = nn.Linear(in_size, 224)  # Adjust input size to match ResNet-18

        # Modify the classifier layers to match the desired output size
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        # # Resize input to match the expected input size of ResNet-18
        # x = self.fc_input(x)

        # Forward pass through the ResNet-18 backbone
        features = self.resnet(x)
        return features


class LinearNN1(nn.Module):
    def __init__(self, in_size=None, hidden_size=None, out_size=None, embed=None, drop_p=0.5, activation='relu', *args,
                 **kwargs):
        super(LinearNN1, self).__init__()
        # super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(in_size, 512)  # Adjust input size to match your data
        self.fc2 = nn.Linear(512, out_size)
        self.dropout = nn.Dropout(p=drop_p)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LinearNN(nn.Module):
    def __init__(self, in_size=None, hidden_size=None, out_size=None, embed=None,
                 drop_p=0.1, activation='softplus'):
        super(LinearNN, self).__init__()
        self.hidden = hidden_size
        hidden = self.hidden
        self.embed = embed
        self.in_size = in_size
        self.out_size = out_size  # number of labels
        self.drop_p = drop_p
        self.activation = activation
        if self.activation == 'softplus':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden),
                nn.Softplus(),
                nn.Linear(hidden, embed),
            )
            self.decoder = nn.Sequential(
                nn.Linear(embed, hidden),
                nn.Softplus(),
                nn.Linear(hidden, hidden),
                nn.Softplus(),
                nn.Linear(hidden, hidden),
                nn.Softplus(),
                nn.Linear(hidden, hidden),
                nn.Softplus(),
                nn.Dropout(p=self.drop_p),
                nn.Linear(hidden, out_size),
                nn.Softplus(),
            )

    def forward(self, data):
        emb = self.encoder(data)
        output = self.decoder(emb)
        return output


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]

        # value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        #  input data consists of simple numerical features rather than sequential data (such as text or time series).
        #  treat each sample as an independent data point without considering sequential properties.
        value_len, key_len, query_len = 1, 1, 1,

        # Split the embedding into self.heads different pieces, [batchsize, sequence_length, num_heads, head_dim]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(queries)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each N
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, self.heads * self.head_dim
        )  # (N, (query_len,) heads, head_dim)

        out = self.fc_out(out)

        # Residual connection and layer normalization
        # out += query
        # out = self.layer_norm(out)
        return out


class LinearNNWithSelfAttention(nn.Module):
    def __init__(self, in_size=None, hidden_size=None, out_size=None, embed=None,
                 drop_p=0.5, activation='relu', num_heads=8):
        super(LinearNNWithSelfAttention, self).__init__()
        self.hidden = hidden_size
        self.embed = embed
        self.in_size = in_size
        self.out_size = out_size  # number of labels
        self.drop_p = drop_p
        self.activation = activation
        self.num_heads = num_heads

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, embed),
        )

        # Self-attention layer
        self.self_attention = SelfAttention(embed, num_heads)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(embed, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, out_size),
            nn.Softmax(dim=-1)  # Softmax along the last dimension
        )

        self.dropout = nn.Dropout(p=self.drop_p)

    def forward(self, data, mask=None):
        emb = self.encoder(data)
        emb = self.dropout(emb)  # Apply dropout after encoding

        # Self-attention mechanism
        self_attended = self.self_attention(emb, emb, emb, mask)

        output = self.decoder(self_attended)
        return output


    def get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leakyrelu':
            return nn.LeakyReLU()
        elif self.activation == 'elu':
            return nn.ELU()
        elif self.activation == 'softplus':
            return nn.Softplus()
        else:
            raise ValueError("Invalid activation function")

# Example usage:
# model = LinearNNWithSelfAttention(in_size=100, hidden_size=64, out_size=10, embed=32)
# input_data = torch.randn(32, 100)  # Example input data, batch size 32, input size 100
# output = model(input_data)
# print(output.shape)  # Should print torch.Size([32, 10]), indicating batch size 32, output size 10
