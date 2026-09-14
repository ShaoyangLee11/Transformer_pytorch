import os
# HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
import torch.optim as optim
from tqdm import tqdm


from datasets import load_dataset
from tokenizers import Tokenizer                        
from tokenizers.models import WordLevel                 
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import BilingualDataset,causal_mask
from model import build_transformer,Transformer
from config import get_weight_file,get_config,get_config_yaml,latest_weights_file_path

from torch.utils.tensorboard import SummaryWriter

def translate(sentence:str):

    stc=sentence
    config=get_config()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #or use:config=get_config_yaml()

    tokenizer_src=Tokenizer.from_file(path=config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt=Tokenizer.from_file(path=config['tokenizer_file'].format(config['lang_tgt']))

    model=build_transformer(tokenizer_src.get_vocab_size(),config['seq_len'],tokenizer_tgt.get_vocab_size(),config['seq_len'],config['d_model'])
    model_latest_weight=latest_weights_file_path(config)
    state=torch.load(model_latest_weight)
    model.load_state_dict(state['model_state_dict'])

    model.eval()
    model=model.to(device)

    with torch.no_grad():
        src=torch.tensor(tokenizer_src.encode(stc).ids,dtype=torch.int64).unsqueeze(0)  #(1,seq_len)
        sos_src=torch.full(size=(1,1),fill_value=tokenizer_src.token_to_id('[SOS]'))    #(1,1)
        eos_src=torch.full(size=(1,1),fill_value=tokenizer_src.token_to_id('[EOS]'))    #(1,1)
        pad_src=torch.full(size=(1,1),fill_value=tokenizer_src.token_to_id('[PAD]'))    #(1,1)

        pad_num=config['seq_len']-src.shape[1]-2

        encoder_input=torch.cat([sos_src,src,eos_src,pad_src*pad_num],dim=-1).to(device)
        encoder_mask=(encoder_input != pad_src.item()).to(device)

        encoder_output=model.encode(encoder_input,encoder_mask).to(device)

        decoder_input=torch.full(size=(1,1),fill_value=tokenizer_tgt.token_to_id('[SOS]')).to(device)    #(1,1)

        while True:
            if decoder_input.shape[1] == config['seq_len']:
                break
            decoder_mask=causal_mask(size=decoder_input.shape[1]).to(device)               # 推理阶段decoder_mask无需&padding_mask
            decoder_output=model.decode(encoder_output,decoder_input,decoder_mask,encoder_mask).to(device)

            prob=model.project(decoder_output[:,-1,:]).to(device)                          # (1,seq_len,d_model)-->(1,1,vocab_size)

            _,next_word=torch.max(prob,dim=-1)

            decoder_input=torch.cat([decoder_input,torch.full(size=(1,1),fill_value=next_word.item()).to(device)],dim=-1)

            if next_word.item()==tokenizer_tgt.token_to_id('[EOS]'):

                break

    
        # decoder_input还在GPU，转cpu再decode
        pred_tokens = decoder_input.detach().cpu().numpy()[0]
        pred_text = tokenizer_tgt.decode(pred_tokens)
        return pred_text

        
if __name__=="__main__":
    while True:
        sentence = input("\n Typing(press 'q' to quit): ").strip()
        if sentence.lower() == "q":
            break
        res = translate(sentence)
        print("Translation:", res)


