import json


data = {}
data['n_context_sent'] = 2
data['train_dataset'] = "WIKIPEDIA_727K" # NEWS_ARTICLES or WIKIPEDIA_727K
data['wiki_datapath'] = "/home/aakash/amagi/data/wiki/wiki_727/"
data['news_datapath'] = "/home/aakash/amagi/data/News_articles/articles/"

data['wiki_batch_size'] = 2
data['news_batch_size'] = 32

if data['train_dataset'] == "WIKIPEDIA_727K":
    data['data_path'] = data['wiki_datapath']
    data['batch_size'] = data['wiki_batch_size']
    
elif data["train_dataset"] == "NEWS_ARTICLES":
    data['data_path'] = data['news_datapath']
    data['batch_size'] = data['news_batch_size']

with open('data_config.json', 'w') as outfile:
    json.dump(data, outfile)
    
    
model_config = {}

model_config["n_classes"] = 2
model_config["learning_rate"] = 0.0005
model_config["class_weights"] = [0.1,1.0]
model_config["available_gpus"] = "cuda:1"
model_config['freezed_epochs'] = 1
model_config['unfreezed_epochs'] = 2
model_config['early_stop_freezed'] = True

model_config['madel_save_path'] = "/home/aakash/amagi/segmentation/saved_models/"
    
with open('model_config.json', 'w') as outfile:
    json.dump(model_config, outfile)