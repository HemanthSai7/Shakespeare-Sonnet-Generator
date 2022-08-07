import time
import torch
from transformers import get_linear_schedule_with_warmup
from tqdm.notebook import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Engine:
    
    def __init__(self, model, device, config, save_file_name = 'model_weights', weight_path='./'):
        
        self.train_loss=dict()
        self.valid_loss=dict()
        self.model=model
        self.device=device
        self.config=config
        self.best_score=0
        self.best_loss=5000
        self.save_file_name = save_file_name
        self.weight_path = weight_path

    def fit(self, train_loader, valid_loader):

      num_train_steps = int(len(train_loader) / self.config.batch_size * self.config.epochs)
      self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
      self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
      
      for epoch in range(self.config.epochs):
        
        t=time.time()
        print("Training Started...")
        
        summary_loss = self.train_one_epoch(train_loader)
        self.train_loss[epoch] = summary_loss.avg

        print(f'Train : Epoch {epoch}: | Summary Loss: {summary_loss.avg} | Training time: {time.time() - t}')
            
        t=time.time()
        print("Validation Started...")
        
        summary_loss = self.validation(valid_loader)
        self.valid_loss[epoch] = summary_loss.avg

        print(f'Valid : Epoch {epoch}: | Summary Loss: {summary_loss.avg} | Training time: {time.time() - t}')
        
        if not self.best_score:
            self.best_score = summary_loss.avg
            print(f'Saving model with lowest validation loss as {self.best_score}')
            self.model.eval()   
            patience = self.config.patience
            torch.save({'model_state_dict': self.model.state_dict(),'best_score': self.best_score, 'epoch': epoch},  f"{self.weight_path}/{self.save_file_name}.pt")
            continue  

        if summary_loss.avg <= self.best_score:
            self.best_score = summary_loss.avg
            patience = self.config.patience  
            print('Improved model with lowest validation loss as {}'.format(self.best_score))
            torch.save({'model_state_dict': self.model.state_dict(),'best_score': self.best_score, 'epoch': epoch},  f"{self.weight_path}/{self.save_file_name}.pt")
        else:
            patience -= 1
            print('Patience Reduced')
            if patience == 0:
                print(f'Early stopping. Lowest validation loss achieved: {self.best_score}')
                break

    def train_one_epoch(self, train_loader):
      self.model.train()

      t = time.time()
      summary_loss = AverageMeter()
      
      for steps, data in enumerate(tqdm(train_loader)):
          ids = data["ids"]
          mask = data["mask"]
          labels = data['target']

          ids = ids.to(self.device, dtype=torch.long)
          mask = mask.to(self.device, dtype=torch.long)
          labels = labels.to(self.device,dtype=torch.long)
            
          self.optimizer.zero_grad()
          outputs = self.model(
              input_ids =ids,
              attention_mask=mask,
              labels = labels
          )

          loss, logits = outputs[:2]                        
          loss.backward()

          self.optimizer.step()
          self.scheduler.step()

          summary_loss.update(loss.detach().item(), self.config.batch_size)

      return summary_loss

    def validation(self, valid_loader):
      self.model.eval()

      t = time.time()
      summary_loss = AverageMeter()

      with torch.no_grad():
        for steps, data in enumerate(tqdm(valid_loader)):
            ids = data["ids"]
            mask = data["mask"]
            labels = data['target']

            ids = ids.to(self.device, dtype=torch.long)
            mask = mask.to(self.device, dtype=torch.long)
            labels = labels.to(self.device,dtype=torch.long)
              
            outputs = self.model(
                input_ids =ids,
                attention_mask=mask,
                labels = labels
            )

            loss, logits = outputs[:2]  
            summary_loss.update(loss.detach().item(), self.config.batch_size) 
      return summary_loss