import torch
import torch.nn as nn
from utils.metric import Metric_hist, History_keeper
import time


class Test:
    def __init__ (  self,
                    test_dataloader,
                    device,
                    criteria):
        
        self.test_dataloader=test_dataloader
        self.device=device
        self.criteria=criteria
    
    def run(self, model,epoch):
        model.eval()
        self.test_loss_hist = Metric_hist()
        
        with torch.no_grad():
            for data, target, time_index in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward
                out = model(data)
                loss=self.criteria(out,target)
                self.test_loss_hist.add_value(float(loss),len(target))
        

        total_loss  = self.test_loss_hist.get_metric
        print(f"\nTEST RESULTS: Epoch: {epoch},loss: {total_loss}")
        
        return total_loss