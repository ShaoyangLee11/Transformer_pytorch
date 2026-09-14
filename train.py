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
from model import build_transformer
from config import get_weight_file,get_config,get_config_yaml

from torch.utils.tensorboard import SummaryWriter

def get_all_sentences(ds,lang):

    for item in ds:
        yield item['translation'][lang]

    """
    ds结构类似于:
    ds={
        item = {
            "translation": {
                "en": "I love AI",
                "zh": "我喜欢人工智能"
            }
        }
        ...
    }
    yield 会把这个函数变成一个生成器函数(generator)
    普通函数用 return : 一次性返回结果，函数直接结束。
    生成器函数用 yield : 产出一个值，暂停函数执行；下次迭代的时候，从暂停位置继续往下跑

    """

def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path=Path(config['tokenizer_file'].format(lang))

    """
    config在此处是一个字典,用来存储模型配置,例如：

    config = {
        "tokenizer_file": "tokenizer_{0}.json",
        # ...其他参数
    }

    config['tokenizer_file']对应的值就是一个模板字符串,也就是:"tokenizer_{0}.json"

    {0}是占位符，用来后面填充变量

    .format()是字符串内置函数,其作用把变量 lang 填充进字符串模板的{0}位置

    假设lang='en',那么config['tokenizer_file'].format(lang)对应的值就是"tokenizer_en.json"

    Path("tokenizer_en.json")的作用就是将"tokenizer_en.json"转化成路径对象

    """

    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer=Whitespace()
        trainer=WordLevelTrainer(special_tokens=['[UNK]','[PAD]','[SOS]','[EOS]'],min_frequency=2)
        tokenizer.train_from_iterator(iterator=get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):

    ds_raw=load_dataset('Helsinki-NLP/opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train',cache_dir="/root/autodl-tmp/hf_cache")

    tokenizer_src:Tokenizer=get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt:Tokenizer=get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    # 90% for training 10% for validation

    train_ds_size=int(0.9*len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size

    train_ds_raw,val_ds_raw=random_split(ds_raw,lengths=[train_ds_size,val_ds_size])

    train_ds=BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds=BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])

    max_len_src=0
    max_len_tgt=0

    for item in ds_raw:
        src_ids=tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids=tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src=max(max_len_src,len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))

    print(f'Max length of source sentence:{max_len_src}')
    print(f'Max length of target sentence:{max_len_tgt}')

    train_dataloader=DataLoader(train_ds,config['batch_size'],shuffle=True)
    val_dataloader=DataLoader(val_ds,batch_size=1,shuffle=False)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt

def get_model(config,src_vocab_len,tgt_vocab_len):

    model=build_transformer(src_vocab_len,config['seq_len'],tgt_vocab_len,config['seq_len'],config['d_model'])
    return model


def train(config):

    # choose device
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    """
    parents=True:假设你希望创建model_folder/exp但传入的字符串是output/model_folder/exp,parent=True会帮你自动创建output父级目录

    exist_ok=True:假如model_folder目录已经存在,exist_ok=True就不会报错,反之报错

    """

    # 加载dataloader
    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)

    # 加载模型
    model=get_model(config=config,src_vocab_len=tokenizer_src.get_vocab_size(),tgt_vocab_len=tokenizer_tgt.get_vocab_size()).to(device)

    writer=SummaryWriter(config['experiment_name'])

    """
    创建 TensorBoard 日志写入器，日志文件就存放在 config['experiment_name'] 这个文件夹下,
    用来记录训练过程数据,之后用 tensorboard 命令打开网页可视化看 loss、精度、图片、模型图等
    日志里面存的是:step、loss 值、图片、直方图等结构化数据
    """

    optimizer=optim.Adam(params=model.parameters(),lr=config['lr'])

    initial_epoch=0
    global_step=0

    if config['preload']:
        model_filename=get_weight_file(config=config,epoch=config['num_epochs'])
        print(f'Preload model {model_filename}')
        state=torch.load(model_filename)
        initial_epoch=state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step=state['global_step']

    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config['num_epochs']):

        model.train()
        batch_iterator=tqdm(iterable=train_dataloader,desc=f'Processing epoch{epoch:02d}')
        """
        train_dataloader是iterable,也就是可迭代对象
        tqdm把train_dataloader包装为迭代器iterator
        iterator存储train_dataloader中的所有数据,但每次迭代只输出一条数据(在dataloader中"一条数据"是指已经被打包成一个batch的数据,所以一次迭代就输出一个batch)

        """
        for batch in batch_iterator:

            encoder_input=batch['encoder_input'].to(device) # (B,seq_len) 
            decoder_input=batch['decoder_input'].to(device) # (B,seq_len)
            encoder_mask=batch['encoder_mask'].to(device)   # (1,1,seq_len)
            decoder_mask=batch['decoder_mask'].to(device)   # (1,seq_len,seq_len)

            encoder_output=model.encode(src=encoder_input,src_mask=encoder_mask)                # (B,seq_len,d_model)
            decoder_ouput=model.decode(encoder_output,decoder_input,decoder_mask,encoder_mask)  # (B,seq_len,d_model)
            proj_output=model.project(decoder_ouput)                                            # (B,seq_len,vocab_size)

            label=batch['label'].to(device)   # (B,seq_len)

            loss=loss_fn(proj_output.reshape(-1,tokenizer_tgt.get_vocab_size()),label.reshape(-1))
            """
            p_o:                                        label:

            [0.1,0.02,0.04,...,0.1]                     [43]
            [0.12,0.003,0.01,...,0.05]                  [12]
            [0.5,0.01,0.01,...,0]                       [500]
            .                                           .
            .                                           .
            .                                           .
            [0.03,0.02,0.1,...,0.6]                     [6002]

            一共B*seq_len个                              一共B*seq_len个
            
            这样就可以计算损失,并非通过广播机制沿着vocab_size维度广播计算
            """

            batch_iterator.set_postfix(loss=f'{loss.item():6.3f}')

            """
            set_postfix 作用:
            在进度条末尾追加一组键值对，实时显示指标(这里就是 loss)每一轮 batch 都会刷新。
            """

            # Log the loss

            writer.add_scalar(tag="train loss",scalar_value=loss.item(),global_step=global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step+=1

        # Save the model after each epoch
        model_filename=get_weight_file(config=config,epoch=f'{epoch:02d}')
        torch.save(
            {
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':model.state_dict(),
                'global_step':global_step
            }
            ,model_filename
        )
if __name__=='__main__':

    config=get_config()
    train(config)


