import itertools
from pprint import pprint
import bz2, json, pickle

import numpy as np
import torch
from transformers import BertTokenizer, BertForNextSentencePrediction
from transformers import XLNetTokenizer, XLNetLMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import requests
from tqdm.autonotebook import tqdm

convai1_data = requests.get('http://convai.io/2017/data/train_full.json').json()
print(len(convai1_data))
convai2_data = requests.get('http://convai.io/data/summer_wild_evaluation_dialogs.json').json()
print(len(convai2_data))

for dial in tqdm(convai1_data):
    quality = sum([participant_score['quality'] for participant_score in dial['evaluation']]) / len(dial['evaluation'])
    dial['quality'] = quality
    utterances = [thread_line['text'] for thread_line in dial['thread']]
    dial['utterances'] = utterances
    dial['predictions'] = dict()
    dial['id'] = str(dial['dialogId'])

convai1_data = [dial for dial in convai1_data if len(dial['utterances']) > 2]
print(len(convai1_data))

for dial in tqdm(convai2_data):
    dial['quality'] = dial['eval_score']
    utterances = [thread_line['text'] for thread_line in dial['dialog']]    
    dial['utterances'] = utterances
    dial['predictions'] = dict()
    dial['id'] = str(dial['dialog_id'])

convai2_data = [dial for dial in convai2_data if len(dial['utterances']) > 2]
print(len(convai2_data))

convai_data_len = convai1_data + convai2_data


######################################################
### Compute BERT NSP scores
######################################################
for BERT_MODEL in tqdm(['bert-base-uncased', 'bert-large-uncased']):

    model_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model = BertForNextSentencePrediction.from_pretrained(BERT_MODEL)

    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()

    for dial in tqdm(itertools.chain(convai1_data, convai2_data), total=convai_data_len):
        utterances = dial['utterances']

        scores = list()

        for u1, u2 in zip(utterances[:-1], utterances[1:]):
            sequence_data = model_tokenizer.encode_plus(text=u1, text_pair=u2, add_special_tokens=True, max_length=512)
            input_ids = sequence_data['input_ids']
            token_type_ids = sequence_data['token_type_ids']
            del sequence_data

            if len(input_ids) > 512:
                print('Sequence too long')

            input_ids = torch.LongTensor(input_ids)
            token_type_ids = torch.LongTensor(token_type_ids)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()

            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)

            with torch.no_grad():
                score = model(input_ids=input_ids, token_type_ids=token_type_ids)[0]
                score = torch.softmax(score, dim=-1)
            score = score.detach().cpu().numpy().squeeze().tolist()
            scores.append(score)

        score_0, score_1 = zip(*scores)
        dial['predictions'][BERT_MODEL+'_nsp_0'] = score_0
        dial['predictions'][BERT_MODEL+'_nsp_1'] = score_1


######################################################
### XLNet scoring function
######################################################

# https://github.com/huggingface/transformers/issues/917#issuecomment-525297746
def xlnet_sent_probability(PADDING_TEXT, text):
    tokenize_text = model_tokenizer.tokenize(text)[:512]
    tokenize_input = model_tokenizer.tokenize(PADDING_TEXT)[:511] + ['<eod>'] + tokenize_text

    sentence_word_probs = list()
    sentence_best_word_probs = list()
    best_words = list()
    # (num_words, num_layers, num_heads, sequence_length, sequence_length
    words_att_1 = torch.zeros((
        max(1, len(tokenize_text)),
        model.config.n_layer,
        model.config.n_head,
        len(tokenize_input),
        len(tokenize_input)
    ), dtype=torch.float).cpu()
    words_att_2 = torch.zeros((
        max(1, len(tokenize_text)),
        model.config.n_layer,
        model.config.n_head,
        len(tokenize_input),
        len(tokenize_input)
    ), dtype=torch.float).cpu()

    for query_word_idx, max_word_id in enumerate(range((len(tokenize_input)-len(tokenize_text)), (len(tokenize_input)))):

        sent = tokenize_input[:]

        input_ids = torch.tensor([model_tokenizer.convert_tokens_to_ids(sent)])

        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
        perm_mask[:, :, max_word_id:] = 1.0 

        target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)
        target_mapping[0, 0, max_word_id] = 1.0

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            perm_mask = perm_mask.cuda()
            target_mapping = target_mapping.cuda()

        with torch.no_grad():
            predicted_prob = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)[0]
            predicted_prob = torch.softmax(predicted_prob[0][-1], dim=-1)

        predicted_prob = predicted_prob.detach().cpu().numpy()

        word_id = model_tokenizer.convert_tokens_to_ids([tokenize_input[max_word_id]])[0]
        word_prob = predicted_prob[word_id].item()
        best_word_prob = predicted_prob.max().item()

        sentence_word_probs.append(word_prob)
        sentence_best_word_probs.append(best_word_prob)
        best_words.append(model_tokenizer.convert_ids_to_tokens(predicted_prob.argmax().item()))

    return (sentence_word_probs, sentence_best_word_probs, best_words)

######################################################
### Compute XLNet scores
######################################################

for XLNET_MODEL in tqdm(['xlnet-base-cased', 'xlnet-large-cased']):

    model_tokenizer = XLNetTokenizer.from_pretrained(XLNET_MODEL)
    model = XLNetLMHeadModel.from_pretrained(XLNET_MODEL)

    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()

    for dial in tqdm(itertools.chain(convai1_data, convai2_data), total=convai_data_len):
        utterances = dial['utterances']

        sentences_word_probs = list()
        sentences_best_word_probs = list()
        sentences_best_words = list()

        for u1, u2 in zip(utterances[:-1], utterances[1:]):
            try:
                sentence_word_probs, sentence_best_word_probs, best_words = xlnet_sent_probability(u1, u2)

                sentences_word_probs.append(sentence_word_probs)
                sentences_best_word_probs.append(sentence_best_word_probs)
                sentences_best_words.append(best_words)
            except Exception as ex:
                print(ex)

        dial['predictions'][XLNET_MODEL+'_word_probs'] = sentences_word_probs
        dial['predictions'][XLNET_MODEL+'_best_word_probs'] = sentences_best_word_probs
        dial['predictions'][XLNET_MODEL+'_best_words'] = sentences_best_words


######################################################
### GPT2 scoring function
######################################################

def gpt2_sent_probability(PADDING_TEXT, text):
    
    tokenize_text = model_tokenizer.encode(text, add_special_tokens=False)[:512]
    tokenize_input = [model_tokenizer.bos_token_id] + \
        model_tokenizer.encode(PADDING_TEXT, add_special_tokens=False)[:510] + tokenize_text + \
        [model_tokenizer.eos_token_id]
    tokenize_text = tokenize_text + \
        [model_tokenizer.eos_token_id]
    tokenize_text_len = len(tokenize_text)

    tokenize_input = torch.LongTensor(tokenize_input)

    if torch.cuda.is_available():
        tokenize_input = tokenize_input.cuda()

    with torch.no_grad():
        predicted_probs = model(tokenize_input)[0]
        predicted_probs = predicted_probs[-tokenize_text_len:-1]
        predicted_probs = torch.softmax(predicted_probs, dim=-1)

    predicted_probs = predicted_probs.detach().cpu().numpy().tolist()
    
    sentence_word_probs = list()
    sentence_best_word_probs = list()
    best_words = list()

    for predicted_prob, token_id in zip(predicted_probs, tokenize_text):
        sentence_word_probs.append(predicted_prob[token_id])
        max_prob = max(predicted_prob)
        sentence_best_word_probs.append(max_prob)        
        best_words.append(
            model_tokenizer.convert_ids_to_tokens(
                predicted_prob.index(max_prob)))

    return (sentence_word_probs, sentence_best_word_probs, best_words)


######################################################
### Compute GPT2 scores
######################################################

for GPT2_MODEL in tqdm(['gpt2', 'gpt2-medium', 'gpt2-large']):

    model_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL)
    model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL, output_past=False)

    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()

    for dial in tqdm(itertools.chain(convai1_data, convai2_data), total=convai_data_len):
        utterances = dial['utterances']

        sentences_word_probs = list()
        sentences_best_word_probs = list()
        sentences_best_words = list()

        for u1, u2 in zip(utterances[:-1], utterances[1:]):
            try:
                sentence_word_probs, sentence_best_word_probs, best_words = gpt2_sent_probability(u1, u2)

                sentences_word_probs.append(sentence_word_probs)
                sentences_best_word_probs.append(sentence_best_word_probs)
                sentences_best_words.append(best_words)
            except Exception as ex:
                print(ex)

        dial['predictions'][GPT2_MODEL+'_word_probs'] = sentences_word_probs
        dial['predictions'][GPT2_MODEL+'_best_word_probs'] = sentences_best_word_probs
        dial['predictions'][GPT2_MODEL+'_best_words'] = sentences_best_words


with bz2.open('./convai1_results.pickle.bz2', mode='wb') as fout:
    pickle.dump(obj=convai1_data, fp=fout)

with bz2.open('./convai2_results.pickle.bz2', mode='wb') as fout:
    pickle.dump(obj=convai2_data, fp=fout)