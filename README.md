# LaBSE Pytorch Model

Pytorch model of LaBSE from [Language-agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852) by Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang of Google AI.

## Abstract from the paper

> We adapt multilingual BERT to produce language-agnostic sen- tence embeddings for 109 languages. While English sentence embeddings have been obtained by fine-tuning a pretrained BERT model, such models have not been applied to multilingual sentence embeddings. Our model combines masked language model (MLM) and translation language model (TLM) pretraining with a translation ranking task using bi-directional dual encoders. The resulting multilingual sentence embeddings improve average bi-text retrieval accuracy over 112 languages to 83.7%, well above the 65.5% achieved by the prior state-of-the-art on Tatoeba. Our sentence embeddings also establish new state-of-the-art results on BUCC and UN bi- text retrieval.

## Convert to Pytorch model
You can directly download the Pytorch model or convert the Tensorflow model by yourself.
### Download the Pytorch model
You can download the Pytorch model from Google Drive: https://drive.google.com/drive/folders/1AmWfiCUhSGgjB_v2Bxcldu466Q2RG8o7?usp=sharing
### Convert by yourself
- A Tensorflow model is available on [Tensorflow Hub](https://tfhub.dev/google/LaBSE/1). 
- Download LaBSE Tensorflow Bert model files from [bojone/labse](https://github.com/bojone/labse#language-agnostic-bert-sentence-embedding-labse).
- Install [huggingface/transformers](https://github.com/huggingface/transformers) and convert the Tensorflow model to Pytorch model using [transformers-cli](https://huggingface.co/transformers/converting_tensorflow_models.html), point `BERT_BASE_DIR` to the downloaded Tensorflow Bert model files, and run:
```bash
export BERT_BASE_DIR=/path/to/labse_tensorflow_bert/labse_bert_model
transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
      --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin
cp ../data/labse_bert_model/bert_config.json ../data/labse_bert_model/config.json 
```

## Use the Pytorch model
Code to reproduce the example from the original [Tensorflow Hub](https://tfhub.dev/google/LaBSE/1):
```python
from transformers import *
import torch
from sklearn.preprocessing import normalize

tokenizer = AutoTokenizer.from_pretrained("../data/labse_bert_model", do_lower_case=False)
model = AutoModel.from_pretrained("../data/labse_bert_model")

max_seq_length = 64
english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]

tok_res = tokenizer(english_sentences, add_special_tokens=True, padding='max_length', max_length=max_seq_length)
input_ids = tok_res['input_ids']
token_type_ids = tok_res['token_type_ids']
attention_mask = tok_res['attention_mask']

device=torch.device('cpu')
# device=torch.device('cuda') # uncomment if using gpu
input_ids = torch.tensor(input_ids).to(device)
token_type_ids = torch.tensor(token_type_ids).to(device)
attention_mask = torch.tensor(attention_mask).to(device)
model = model.to(device)

output = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
english_embeddings = output[1].cpu().detach().numpy()
english_embeddings = normalize(english_embeddings)
```
Notice that the embeddings are l2-normalized in the paper as well as the Tensorflow Hub implementation, and that is what we did at the end of the code.

