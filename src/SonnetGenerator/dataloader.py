import torch
from torch.utils.data import Dataset

class SonnetDataset(Dataset):
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        self.eos_tok="<|endoftext|>"       
        self.sonnets=[] 

        with open('../data/Sonnets.txt') as txt_file:
          sonnett=txt_file.lower().readlines()

          for line in sonnett:
            sonnet=f"Sonnet: {str(line)}{self.eos_tok}"
            self.sonnets.append(sonnet)


    def __getitem__(self,idx):
        sonnet=self.sonnets[idx]
        
        inputs=self.tokenizer.encode_plus(
            sonnet,
            None,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            max_length=256,
            )

        ids=inputs["input_ids"]
        mask=inputs["attention_mask"]


        return {"ids":torch.tensor(ids,dtype=torch.long),
                "mask":torch.tensor(mask,dtype=torch.long),
                "target":torch.tensor(ids,dtype=torch.long)}   
    
    def __len__(self):
        return len(self.sonnets)