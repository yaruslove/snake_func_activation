import numpy as np

class Metric_hist:
    def __init__(self):
        self.hist = np.empty((0,2), np.half)
        
    def add_value(self, acc, amount_samples):
        self.hist = np.append(self.hist, np.array([[acc,amount_samples]]), axis=0)
    
    @property
    def get_metric(self):
        val_data=np.sum(self.hist[:,0:1]*self.hist[:,1:2],axis=0)/np.sum(self.hist[:,1:2])
        return round(float(val_data),4)



class History_keeper:
    def __init__ (self):
        self.test_hist_loss=[]
        # self.test_hist_accuracy=[]
        self.train_hist_loss=[]
        # self.train_hist_accuracy=[]
    
    def add_test(self, loss, epoch): # accur
        self.test_hist_loss.append(loss)
        # self.test_hist_accuracy.append(accur)

        
    def add_train(self, loss ,epoch): # accur
        self.train_hist_loss.append(loss)
        # self.train_hist_accuracy.append(accur)