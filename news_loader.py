import pandas as pd
from torch.utils.data import Dataset

def read_articles(article):
    special_symbols = {'J.': '$JTrumph$', 'Dr.':'$dr$', 'Mr.':'$mr$', 'Mrs.':'$mrs$'}
    text = []
    
    temp = article.replace('J.', '$JTrumph$')
    temp = temp.replace('Dr.', '$dr$')
    temp = temp.replace('Mr.', '$mr$')
    temp = temp.replace('Mrs.', '$mrs$')
    temp = temp.replace('U.S.', '$us$')
    temp = temp.replace(". ", ".\n").split("\n")
    
    for sent in temp: 
        sent_ = sent.replace( '$JTrumph$', 'J.')
        sent_ = sent_.replace( '$dr$', 'Dr.')
        sent_ = sent_.replace( '$mr$','Mr.')
        sent_ = sent_.replace('$mrs$', 'Mrs.')
        sent_ = sent_.replace('$us$', 'U.S.')
        text.append(sent_)
        
    return text
            
class NewsDataset(Dataset):
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        return read_articles(self.df['content'][idx])