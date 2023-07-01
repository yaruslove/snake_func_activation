import torch
import torch.nn as nn
from utils.metric import Metric_hist, History_keeper
import time


class Train:
    def __init__ (self,
                  train_dataloader: torch.utils.data.dataloader.DataLoader,
                  device: torch.device,
                  optimizer: torch.optim,
                  criteria: nn.Module,
                  scheduler):
    
        self.train_dataloader=train_dataloader
        self.device=device
        self.optimizer=optimizer
        self.criteria=criteria
        self.scheduler=scheduler
        
        self.amount_batches = len(train_dataloader)

    def run(self,model,epoch):
        self.train_loss_hist = Metric_hist()
        
        model.train()
        
        now = time.time()
        for idx, (data, target, time_index) in enumerate(self.train_dataloader):
            # Prepare data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            out = model(data)
            loss=self.criteria(out,target)
            self.train_loss_hist.add_value(float(loss),len(target))
            
            # Backward + step grad
            loss.backward()
            self.optimizer.step()
            
            # Count metric
            print(f'\rRunning during epoch: {epoch}, iteration: {idx} of {self.amount_batches} loss: {loss}',end="")

            self.scheduler.step()
        
        total_loss  = self.train_loss_hist.get_metric
        print(f"\nTRAIN RESULTS: Epoch: {epoch},loss: {total_loss}")
        
        # Counting time 
        print(f"Time difference {int(time.time()) - now}")

        return model, total_loss


