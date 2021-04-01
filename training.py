import re
import os
import math
import gensim
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score
import json

from wiki_loader import WikipediaDataSet
from news_loader import NewsDataset

import transformers
from transformers import AdamW
from transformers import BertModel, BertTokenizer

bert = BertModel.from_pretrained('bert-base-uncased')
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_TOKENS = 200


#### freezing bert ########
for param in bert.parameters():
    param.requires_grad = False
    

def collate_fn_wiki(batch):
    batched_data = []
    batched_targets = []
    batched_paths = []


    max_tokens = 100
    for data, targets, path in batch:
        try:
            for i in range(len(data)):
                temp = len(data[i][0].split())+len(data[i][1].split())
                if max_tokens < temp:
                    max_tokens = temp
                batched_data.append(data[i])
                batched_targets.append(targets[i])
                batched_paths.append(path)
        except Exception as e:
            logger.info('Exception "%s" in file: "%s"', e, path)
            logger.debug('Exception!', exc_info=True)
            continue
    
    max_tokens = min(MAX_TOKENS, max_tokens)
    tokens = tokenizer(
                    batched_data,
                    padding = True,
                    max_length = max_tokens,
                    truncation=True)
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    #y = torch.tensor(batched_targets,dtype=torch.float32).unsqueeze(axis=1)
    y = torch.tensor(targets)    
    return seq, mask, y, batched_paths


def collate_fn_news(batch):

    final_sentences = []
    label = []
    for section_ind in range(len(batch)):
            sentences = batch[section_ind]
            if sentences:
                for sentence in sentences[:-1]:
                    final_sentences.append(sentence)
                    label.append(0)
                final_sentences.append(sentences[-1])
                label.append(1)

    data = []
    targets = []
    if len(final_sentences)>n_context_sent:
            for sent_ind in range(n_context_sent,len(final_sentences)):
                prev_context = final_sentences[sent_ind-n_context_sent:sent_ind]
                after_context = final_sentences[sent_ind: min(len(final_sentences),sent_ind+n_context_sent)]

                prev_context = " ".join(prev_context)
                after_context = " ".join(after_context)
                data.append([prev_context, after_context])
                targets.append(label[sent_ind-1])
                
    max_tokens = 150
    tokens = tokenizer(
                    data,
                    padding = True,
                    max_length = max_tokens,
                    truncation=True)
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    #y = torch.tensor(targets).unsqueeze(axis=1)
    y = torch.tensor(targets)
    return seq, mask, y

with open('data_config.json') as json_file:
    data_config = json.load(json_file)
    dataset_path = data_config['data_path']
    n_context_sent = data_config['n_context_sent']
    dataset_name = data_config['train_dataset']
    batch_size = data_config['batch_size']
    
dataset_train = None
train_dataloader = None
dataset_val = None
val_dataloader = None

if dataset_name == "WIKIPEDIA_727K":
    dataset_train = WikipediaDataSet(dataset_path+'train',n_context_sent= n_context_sent, high_granularity=False)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, collate_fn = collate_fn_wiki, shuffle=True)
    
    dataset_val = WikipediaDataSet(dataset_path+'dev',n_context_sent= n_context_sent, high_granularity=False)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_fn_wiki, shuffle=False)
elif dataset_name == "NEWS_ARTICLES":
    dataset_train = NewsDataset(dataset_path+"train.csv")
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, collate_fn = collate_fn_news, shuffle=True)
    
    dataset_val = NewsDataset(dataset_path+"test.csv")
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_fn_news, shuffle=False)

class Encoder_Classifier(nn.Module):
    def __init__(self, bert, n_classes):
        super(Encoder_Classifier, self).__init__()
        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,64)
        self.out = nn.Linear(64,n_classes)
        
    #define the forward pass
    def forward(self, sent_id, mask):
        #pass the inputs to the model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)


        x = F.relu(self.fc1(cls_hs)) 

        # apply softmax activation
        x = self.out(x)

        return x
        
def train(train_dataloader, is_early=False, max_batches=500):
    model.train()

    total_loss, total_accuracy = 0, 0
    '''
    # empty list to save model predictions
    total_preds=[]'''
    step = 0
    # iterate over batches
    for batch in tqdm(train_dataloader):

        # push the batch to gpu
        #batch = [r.to(device) for r in batch]

        ###### for  labeled data, computing cross entropy   #########
        sent_id, mask, labels = batch[0].to(device),batch[1].to(device), batch[2].to(device)

        model.zero_grad()        
        preds = model(sent_id, mask)
        loss = CELoss(preds, labels)

        # backward pass to calculate the gradients
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()
        
        #torch.cuda.empty_cache()
        # add on to the total loss
        loss_item = loss.item()
        total_loss += loss_item

        # progress update after every 100 batches.
        if step % 50 == 0 and not step == 0:
            
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
            print("loss",loss_item)
            torch.cuda.empty_cache()
            if is_early and step>=max_batches:
                print("early stopping...")
                break
         

        '''
        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)'''
        step+=1
    # compute the training loss of the epoch
    avg_loss = total_loss / step
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    #total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss
    
def evaluate(dev_dataloader, is_early=False, max_batches=500):
  
    print("\nEvaluating...")
  
    # deactivate dropout layers
    model.eval()
    total_loss, total_accuracy = 0, 0
    total_preds = [[None,None]]
    curr_examples = 0

    # iterate over batches
    step = 0
    for batch in tqdm(dev_dataloader):
    
        # Progress update every 10 batches.
        if step % 10 == 0 and not step == 0:
      
            # Report progress.
            temp = np.delete(total_preds,0,0)
            print('  Batch {:>5,}  of  {:>5,} accuracy {}.'.format(step, len(dev_dataloader), accuracy_score(list(temp[:,0]),list(temp[:,1]))))
            
            print("F1 score {}".format(f1_score(list(temp[:,0]),list(temp[:,1]),average="macro")))
            if is_early and step>=max_batches:
                print("early stopping...")
                break
        # push the batch to gpu
        #batch = [t.to(device) for t in batch]
    

        sent_id, mask, labels = batch[0].to(device),batch[1].to(device),batch[2].to(device)
        curr_examples += len(sent_id)
        # deactivate autograd
        with torch.no_grad():
      
        # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = CELoss(preds,labels)

            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            preds_ = []
            preds_ = np.argmax(preds,axis=1)
            '''
            for i in preds:
                if i[0]<=.5:
                    preds_.append(0)
                else:
                    preds_.append(1)'''

            true_class = np.expand_dims(preds_,axis=1)
            #labels = labels.detach().cpu().numpy()
            labels = np.expand_dims(labels.detach().cpu().numpy(),axis=1)
            temp = np.concatenate((labels,true_class),axis=1)
            
            
            total_preds = np.concatenate((total_preds,temp),axis=0)
            
        step+=1
    # compute the validation loss of the epoch
    avg_loss = total_loss / step 

    return avg_loss, total_preds

#### loading model configurations  #####

with open('model_config.json') as json_file:
    model_config = json.load(json_file)
    n_classes = model_config["n_classes"]
    learning_rate = model_config["learning_rate"]
    class_weights = model_config["class_weights"]
    gpu = model_config["available_gpus"]
    freezed_epochs = model_config['freezed_epochs']
    unfreezed_epochs = model_config['unfreezed_epochs']
    model_path = model_config['madel_save_path']
    
#### model training configurations  ######
best_valid_f1 = 0.0
train_losses = []
valid_losses = []
accuracy_list = []
n_layer_unfreeze = 1
device = torch.device(gpu if torch.cuda.is_available() else "cpu")


class_weights = torch.FloatTensor(class_weights).to(device)

CELoss = nn.CrossEntropyLoss(weight= class_weights)

model = Encoder_Classifier(bert, n_classes).float()
#model = nn.DataParallel(model, device_ids=[0, 1])
model.to(device)
optimizer = AdamW(model.parameters(),lr = learning_rate)
print("model will run on: {}".format(device))

##### training starts ########
for epoch in range(freezed_epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, freezed_epochs))
    
    #train model
    train_loss = train(train_dataloader, is_early=True)
    
    #evaluate model
    valid_loss, preds = evaluate(val_dataloader, is_early=True)
    
    temp = np.delete(preds,0,0)
    valid_f1 = f1_score(list(temp[:,0]),list(temp[:,1]),average="macro")
    accuracy = accuracy_score(list(temp[:,0]),list(temp[:,1]))
    #save the best model
    if valid_f1 > best_valid_f1:
        best_valid_f1 = valid_f1
        torch.save(model.state_dict(), model_path+'saved_weights_'+dataset_name+'.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    accuracy_list.append(accuracy)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')


##### unfreezing layers #######
for iter in range(n_layer_unfreeze):
    print(str(iter+1)+" unfreeze")
    for param in model.bert.encoder.layer._modules[str(11-iter)].parameters():
        param.requires_grad=True
        
    for epoch in range(2):
     
        print('\n Epoch {:} / {:}'.format(epoch+1 ,unfreezed_epochs ))
        
        #train model
        train_loss = train(train_dataloader, is_early=True, max_batches=300)
    
        #evaluate model
        valid_loss, preds = evaluate(val_dataloader,is_early=True)
        
        temp = np.delete(preds,0,0)
        valid_f1 = f1_score(list(temp[:,0]),list(temp[:,1]),average="macro")
        print(valid_f1)
        #save the best model
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            torch.save(model.state_dict(), model_path+'saved_weights_'+dataset_name+'.pt')
        
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        accuracy_list.append(accuracy)
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
