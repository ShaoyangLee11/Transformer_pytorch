import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size):
        super().__init__()

        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)

    """
    nn.Embedding 接收 (B, seq_len) 的整数索引
    输出 (B, seq_len, d_model)
    nn.Embedding的本职工作:根据 token 索引查表，把单个整数 id 映射成 d_model 维向量。

    形如(vocab_size,)的独热编码张量在实操中根本不存在,取而代之的是索引
    """

class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        #在此构造一个(seq_len,d_model)的矩阵
        pe=torch.zeros(self.seq_len,self.d_model)
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(dim=1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        pe[::,0::2]=torch.sin(position*div_term)
        pe[::,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0)              #变为(1,seq_len,d_model)

        self.register_buffer('pe',pe)   #并非为模型参数但我希望将其作为参数保存则使用register_buffer()

    def forward(self,x:torch.Tensor):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)     #pe矩阵不是可学习的参数，将requires_grad_()置为False
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self,eps=1e-6):
        super().__init__()

        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))  #使用nn.Parameter()使其成为可学习的参数
        #torch.ones(1)括号里的 `1` 代表张量形状 `(1,)`常量不可前向传播，至少是元组才可以，所以将其定义为(1,)
        self.bias=nn.Parameter(torch.zeros(1))

    def forward(self,x:torch.Tensor):
        mean=x.mean(dim=-1,keepdim=True)        #keepdim不置为True则指定的维度会消失
        std=x.std(dim=-1,keepdim=True)

        return self.alpha*((x-mean)/(std+self.eps))+self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()

        self.linear1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(d_ff,d_model)

    def forward(self,x):

        x=torch.relu(self.linear1(x))
        x=self.linear2(self.dropout(x))

        return x


class MutiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h==0,"d_model is not divisible by h"

        self.d_k=d_model//h

        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)

        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)


    @staticmethod                                           #表示不用创建类示例就可外部调用的方法
    def attention(query,key,value,mask,dropout:nn.Dropout):

        d_k=query.shape[-1]

        # (batch,h,seq_len,d_k)-->(batch,h,seq_len,seq_len)
        attention_scores=(query@key.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores=attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores=dropout(attention_scores)

        return (attention_scores@value),attention_scores


    def forward(self,q,k,v,mask):       #实操中我们传入的q,k,v都是同一个矩阵，也就是词向量堆叠形成的词序列矩阵
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)

        #(batch,seq_len,d_model)-->(batch,seq_len,h,d_k)-->(batch,h,seq_len,d_k)
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)


        x,self.attention_scores=MutiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        #(batch,h,seq_len,d_k)-->(batch,seq_len,h,d_k)-->(batch,seq_len,d_model)
        x=x.transpose(1,2).reshape(x.shape[0],-1,self.d_k*self.h)

        return self.w_o(x)


class ResidualBlock(nn.Module):

    def __init__(self,dropout:float):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MutiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_blocks=nn.ModuleList([ResidualBlock(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x=self.residual_blocks[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))     #使用lambda函数能够很好解决ResidualBlock内部的调用问题
        x=self.residual_blocks[1](x,lambda x:self.feed_forward_block(x))

        return x


class Encoder(nn.Module):

    def __init__(self,layers:nn.ModuleList):            #ModuleList中每一个对象都是EncoderBlock类
        super().__init__()

        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,src_mask):
        for layer in self.layers:
            x=layer(x,src_mask)

        return self.norm(x)
    
"""
src_mask是屏蔽padding用的,在encoder自注意力和decoder交叉注意力的时候用
tgt_mask则是用于屏蔽未来词的,防止当前词看见未来的词,在decoder自注意力的时候用

"""

class DecoderBlock(nn.Module):

    def __init__(self,self_attention_block:MutiHeadAttentionBlock,cross_attention_block:MutiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_blocks=nn.ModuleList([ResidualBlock(dropout=dropout) for _ in range(3)])

    def forward(self,x:torch.Tensor,encoder_output,src_mask,tgt_mask):
        x=self.residual_blocks[0](x,lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_blocks[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_blocks[2](x,lambda x:self.feed_forward_block(x))

        return x
"""
其实从这里可以看出来encoder向交叉注意力输入的k,v矩阵实际上是同一个矩阵
"""

class Decoder(nn.Module):

    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)

        return self.norm(x)

#这一层的作用是将decoder的输出变形:(batch,seq_len,d_model)-->(batch,seq_len,vocab_size)
#这样的作用是将输出映射到此表进行softmax
class ProjectionLayer(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.linear=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        x=self.linear(x)

        return torch.log_softmax(x,dim=-1)

class Transformer(nn.Module):

    def __init__(self,
                 src_embd:InputEmbedding,
                 tgt_embd:InputEmbedding,
                 src_pos:PositionalEncoding,
                 tgt_pos:PositionalEncoding,
                 encoder:Encoder,
                 decoder:Decoder,
                 proj_layer:ProjectionLayer):
        super().__init__()

        self.src_embd=src_embd
        self.src_pos=src_pos

        self.tgt_embd=tgt_embd
        self.tgt_pos=tgt_pos

        self.encoder=encoder
        self.decoder=decoder

        self.proj_layer=proj_layer

    def encode(self,src,src_mask):

        src=self.src_embd(src)
        src=self.src_pos(src)

        return self.encoder(src,src_mask)

    def decode(self,encoder_output,tgt,tgt_mask,src_mask):

        tgt=self.tgt_embd(tgt)
        tgt=self.tgt_pos(tgt)

        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        return self.proj_layer(x)


def build_transformer(src_vocab_size,
                      src_seq_len,
                      tgt_vocab_size,
                      tgt_seq_len,
                      d_model:int=512,
                      N:int=6,
                      h:int=8,
                      dropout:float=0.1,
                      d_ff:int=2048
                      )->Transformer:

    src_embd=InputEmbedding(d_model=d_model,vocab_size=src_vocab_size)
    tgt_embd=InputEmbedding(d_model=d_model,vocab_size=tgt_vocab_size)

    src_pos=PositionalEncoding(d_model=d_model,seq_len=src_seq_len,dropout=dropout)
    tgt_pos=PositionalEncoding(d_model=d_model,seq_len=tgt_seq_len,dropout=dropout)

    encoder_list=[]

    for _ in range(N):
        encoder_self_attention_block=MutiHeadAttentionBlock(d_model=d_model,h=h,dropout=dropout)
        encoder_feed_forward_block=FeedForwardBlock(d_model=d_model,d_ff=d_ff,dropout=dropout)
        encoder_block=EncoderBlock(encoder_self_attention_block,encoder_feed_forward_block,dropout=dropout)

        encoder_list.append(encoder_block)

    encoder=Encoder(nn.ModuleList(encoder_list))

    decoder_list=[]

    for _ in range(N):
        decoder_self_attention_block=MutiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block=MutiHeadAttentionBlock(d_model,h,dropout)
        decoder_feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block=DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,decoder_feed_forward_block,dropout)

        decoder_list.append(decoder_block)

    decoder=Decoder(nn.ModuleList(decoder_list))

    proj_layer=ProjectionLayer(d_model,tgt_vocab_size)

    transformer=Transformer(src_embd,tgt_embd,src_pos,tgt_pos,encoder,decoder,proj_layer)

    for param in transformer.parameters():
        if param.dim()>1:
            nn.init.xavier_uniform(param)

    return transformer

