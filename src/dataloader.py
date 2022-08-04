import torch
from torch.utils.data import Dataset

class SonnetDataset(Dataset):
    def __init__(self,data,tokenizer):
        self.data=data
        self.tokenizer=tokenizer
        self.eos_tok="<|endoftext|>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        sonnet=self.data[idx]
        sonnet=f"Sonnet: {str(sonnet)} {self.eos_tok}"

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