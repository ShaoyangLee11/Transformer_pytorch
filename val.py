import os
# HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
import torch.optim as optim
from tqdm import tqdm

# 此处使用的分词器的逻辑较为简单，基本依据语句中的空格分词

from datasets import load_dataset
from tokenizers import Tokenizer                        #Tokenizer()：tokenizers 库的主类，用来包装分词模型、预分词器、训练器等组件
from tokenizers.models import WordLevel                 #WordLevel():tokenizers.models 库中按词分token的模型
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import BilingualDataset,causal_mask
from model import build_transformer,Transformer
from config import get_weight_file,get_config,get_config_yaml,latest_weights_file_path
from train import get_ds

from torch.utils.tensorboard import SummaryWriter

# encoder的输出可以复用,因此构造greedy_decode()来实现复用
def greedy_decode(model:Transformer,source,source_mask,tokenizer_src:Tokenizer,tokenizer_tgt:Tokenizer,device,max_len):

    sos_idx=tokenizer_tgt.token_to_id('[SOS]')
    eos_idx=tokenizer_tgt.token_to_id('[EOS]')

    encoder_output=model.encode(source,source_mask)

    decoder_input=torch.full(size=(1,1),fill_value=sos_idx).type_as(source).to(device)

    while 1:
        if decoder_input.shape[-1]==max_len:
            break

        decoder_mask=causal_mask(size=decoder_input.shape[-1]).type_as(source).to(device)

        out=model.decode(encoder_output=encoder_output,tgt=decoder_input,tgt_mask=decoder_mask,src_mask=source_mask)

        prob=model.project(out[:,-1,:])

        _,next_word=torch.max(prob,dim=-1)          #torch.max一次返回最大值本身和其对应的索引,因为并不需要最大值本身,使用下划线接收(变相丢弃这个值)并只保留索引

        decoder_input=torch.cat([decoder_input,torch.full(size=(1,1),fill_value=next_word.item())].to(device))

        if next_word.item()==eos_idx:
            break

    return decoder_input.unsqueeze(0)

"""
tip:

每次迭代传给 Decoder 的是「到目前为止已经生成的全部 token 序列」
但我们只用这一轮输出里，最后一个位置的向量，去预测下一个 token。

"""

def run_validation(model:Transformer,validation_ds,device,tokenizer_src,tokenizer_tgt:Tokenizer,max_len,print_msg,num_examples=2):
    model.eval()
    count=0


    #Size of the control window
    console_width=80

    with torch.no_grad():
        for batch in validation_ds:
            count+=1
            encoder_input=batch['encoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)

            decoder_input=batch['decoder_input'].to(device)
            decoder_mask=batch['decoder_mask'].to(device)

            assert encoder_input.shape[0]==1,"Batch size must be 1 for validation"

            model_output=greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,device,max_len)

            src_text=batch['src_text']
            expect=batch['tgt_text']
            pred=tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            print_msg('-'*console_width)
            print_msg(f'SOURCE TEXT:{src_text}')
            print_msg(f'TARGET TEXT:{expect}')
            print_msg(f'PREDICTED TEXT:{pred}')

            if count==num_examples:
                break



if __name__=="__main__":
    config=get_config()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)
    batch_iterator=tqdm(val_dataloader)


    model=build_transformer(tokenizer_src.get_vocab_size(),config['seq_len'],tokenizer_tgt.get_vocab_size(),config['seq_len'])
    model_latest_weight=latest_weights_file_path(config)
    state=torch.load(model_latest_weight)
    model.load_state_dict(state['model_state_dict'])
    model=model.to(device)
    

    run_validation(model, batch_iterator, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg))
