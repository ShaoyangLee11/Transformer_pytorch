#数据集在这里就是一个多级字典，同时每一个token都被分词器赋予id

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split

from datasets import load_dataset
from tokenizers import Tokenizer 

class BilingualDataset(Dataset):

    def __init__(self,ds,tokenizer_src:Tokenizer,tokenizer_tgt:Tokenizer,src_lang,tgt_lang,seq_len):
        super().__init__()
        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt

        self.src_lang=src_lang
        self.tgt_lang=tgt_lang

        self.seq_len=seq_len

        self.sos_token=torch.tensor([tokenizer_src.token_to_id('[SOS]')],dtype=torch.int64)
        self.eos_token=torch.tensor([tokenizer_src.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad_token=torch.tensor([tokenizer_src.token_to_id('[PAD]')],dtype=torch.int64)

    def __len__(self):                          #支持 len(obj)，当你写 len(dataset:BilingualDataset)，Python 自动调用 dataset.__len__() 返回数据集总样本数量
        return len(self.ds)

    def __getitem__(self, index):               #支持下标取值：obj[index]，当你写 dataset[0] ，Python 自动调用 `dataset.__getitem__(0)`

        src_target_pair=self.ds[index]
        src_text=src_target_pair['translation'][self.src_lang]
        tgt_text=src_target_pair['translation'][self.tgt_lang]

        #现在我们需要把文本转化成token在词表中的id,让分词器(tokenizer)使用encode(...).ids会返回一个数组,其中包含语句中各个token的id

        enc_input_tokens=self.tokenizer_src.encode(src_text).ids
        dec_input_tokens=self.tokenizer_tgt.encode(tgt_text).ids

        """
        tip:

        encoder的输入在前后加[SOS]和[EOS]
        训练时decoder的输入只在前加[SOS]
        label只在后加[EOS]

        """

        enc_num_padding_tokens=self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens=self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too short')


        encoder_input=torch.concat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens,dtype=torch.int64)
            ]
        )
        #无需思考形状,以上张量都是一维的

        decoder_input=torch.concat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens,dtype=torch.int64)
            ]
        )



        label=torch.concat(
            [
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens,dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0)==self.seq_len
        assert decoder_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len

        return {
            "encoder_input":encoder_input, #(seq_len,)
            "decoder_input":decoder_input, #(seq_len,)
            "encoder_mask":(encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0), #(1,1,seq_len)
            "decoder_mask":(decoder_input != self.pad_token).unsqueeze(0) & causal_mask(decoder_input.size(0)), #(1,seq_len) & (1,seq_len,seq_len)
            "label":label,
            "src_text":src_text,
            "tgt_text":tgt_text
        }


        """
        tip:

        !=:逐元素比较，生成布尔张量
        - token 不等于 pad → True
        - token 等于 pad → False`

        假设:encoder_input = [ 5, 8, 2, 0, 0 ]
        (encoder_input != self.pad_token)返回的就是[True, True, True, False, False]
        
        """

def causal_mask(size):
    # torch.triu()会保留矩阵上三角的所有值，其余位置全部变为0.
    mask=torch.triu(torch.ones(size=(1,size,size)),diagonal=1).type(torch.int)
    # 现在是一个上三角全为1,其余位置为0的矩阵

    return mask==0
    # 返回布尔矩阵也就是掩码，我们需要上三角全为False的布尔矩阵，因此用mask==0



"""
tip:

在初始数据集是我们只用考虑一条数据即可,无需考虑批次维度

因为包装成batch是dataloader的任务,不是dataset的任务

"""