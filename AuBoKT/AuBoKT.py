

import math
import logging
import torch
import torch.nn as nn
from sklearn import metrics
import tqdm
import numpy as np
import random
import copy

from AuBoKTNet import AuBoKTNet

device = torch.device('cuda' if torch.cuda.is_available else 'cpu' )

def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0-target)*np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0

def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)

def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)

def train_one_epoch(net,begin_idx,retain_h,retain_lstm_h,retain_lstm_c,optimizer, criterion, batch_size, e_data, ed_data,  a_data,sid_data,it_data,pid_try_data,stamp_try_data):
    net.train()
    n = int(math.ceil(len(e_data) / batch_size ))
    shuffled_ind = np.arange(e_data.shape[0])
    
    np.random.shuffle(shuffled_ind)
    e_data = e_data[shuffled_ind]
    ed_data = ed_data[shuffled_ind]
    a_data = a_data[shuffled_ind]
    sid_data = sid_data[shuffled_ind]
    it_data = it_data[shuffled_ind]
    pid_try_data = pid_try_data[shuffled_ind]
    stamp_try_data = stamp_try_data[shuffled_ind]
    
    pred_list = []
    target_list = []
    
    for idx in tqdm.tqdm(range(n), 'Training'):
        optimizer.zero_grad()
        
        e_one_seq = e_data[idx *batch_size :(idx+1) * batch_size, :]
        ed_one_seq = ed_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq  = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        sid_one_seq = sid_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq  = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        pid_try_one_seq  = pid_try_data[idx * batch_size: (idx + 1) * batch_size, :]
        stamp_try_one_seq  = stamp_try_data[idx * batch_size: (idx + 1) * batch_size, :]
        
        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_ed = torch.from_numpy(ed_one_seq).long().to(device)
        input_a = torch.from_numpy(a_one_seq).long().to(device)
        input_sid = torch.from_numpy(sid_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        input_pid_try= torch.from_numpy(pid_try_one_seq).long().to(device)
        input_stamp_try= torch.from_numpy(stamp_try_one_seq).long().to(device)
        
        target = torch.from_numpy(a_one_seq).float().to(device)

        pred = net(input_e, input_ed, input_a, input_sid, input_it,input_pid_try,input_stamp_try,deliver_h=None, deliver_lstm_h=None,deliver_lstm_c=None, train_epoch = True,deliver_h_list=retain_h,deliver_lstm_h_list=retain_lstm_h,deliver_lstm_c_list=retain_lstm_c,begin_idx = 0)

        mask = input_e[:, 1:]>0
        masked_pred = pred[:, 1:][mask]
        masked_truth = target[:, 1:][mask]
        
        loss = criterion(masked_pred, masked_truth).sum()
        loss.backward()
        optimizer.step()
        
        masked_pred = masked_pred.detach().cpu().numpy() 
        masked_truth = masked_truth.detach().cpu().numpy()
        
        pred_list.append(masked_pred)
        target_list.append(masked_truth)
    
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    
    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy, shuffled_ind

def test_one_epoch(net, begin_idx,retain_h,retain_lstm_h,retain_lstm_c,batch_size, shuffled_ind,e_data, ed_data, a_data,sid_data,it_data,pid_try_data,stamp_try_data):
    net.eval()
    n = int(math.ceil(len(e_data) / batch_size))

    pred_list = []
    target_list = []
    
    e_data = e_data[shuffled_ind]
    ed_data = ed_data[shuffled_ind]
    a_data = a_data[shuffled_ind]
    sid_data = sid_data[shuffled_ind]
    it_data = it_data[shuffled_ind]
    pid_try_data = pid_try_data[shuffled_ind]
    stamp_try_data = stamp_try_data[shuffled_ind]

    for idx in tqdm.tqdm(range(n), 'Testing'):
        e_one_seq = e_data[idx *batch_size :(idx+1) * batch_size, :]
        ed_one_seq = ed_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq  = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        sid_one_seq  = sid_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq  = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        pid_try_one_seq  = pid_try_data[idx * batch_size: (idx + 1) * batch_size, :]
        stamp_try_one_seq  = stamp_try_data[idx * batch_size: (idx + 1) * batch_size, :]
        
        retain_h_one_seq = retain_h[idx * batch_size: (idx + 1) * batch_size, :]
        retain_lstm_h_one_seq = retain_lstm_h[idx * batch_size: (idx + 1) * batch_size, :]
        retain_lstm_c_one_seq = retain_lstm_c[idx * batch_size: (idx + 1) * batch_size, :]

        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_ed = torch.from_numpy(ed_one_seq).long().to(device)
        input_a = torch.from_numpy(a_one_seq).long().to(device)
        input_sid = torch.from_numpy(sid_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        input_pid_try = torch.from_numpy(pid_try_one_seq).long().to(device)
        input_stamp_try = torch.from_numpy(stamp_try_one_seq).long().to(device)

        target = torch.from_numpy(a_one_seq).float().to(device)

        with torch.no_grad():
            pred = net(input_e, input_ed, input_a, input_sid, input_it,input_pid_try,input_stamp_try,deliver_h=retain_h_one_seq,deliver_lstm_h=retain_lstm_h_one_seq ,deliver_lstm_c=retain_lstm_c_one_seq,train_epoch = False, deliver_h_list=[],deliver_lstm_h_list=[],deliver_lstm_c_list=[],begin_idx = begin_idx)
            
            mask = input_e[:, begin_idx+1:] > 0
            masked_pred = pred[:, begin_idx+1:][mask].detach().cpu().numpy()
            masked_truth = target[:, begin_idx+1:][mask].detach().cpu().numpy()

            pred_list.append(masked_pred)
            target_list.append(masked_truth)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    
    return loss, auc, accuracy


class AuBoKT(nn.Module):
    def __init__(self, n_e, n_k, n_l, n_sid, n_it, n_pid_try, n_stamp_try, d_e, d_k, d_l, d_a, d_x, d_p, d_it, d_try, q_matrix, q_matrix_i, q_matrix_d, batch_size, u1, sigma1, h1, u2, sigma2, h2,dropout = 0.2):
        super(AuBoKT, self).__init__()
        
        q_matrix = torch.from_numpy(q_matrix).float().to(device)
        q_matrix_i = torch.from_numpy(q_matrix_i).long().to(device)
        q_matrix_d = torch.from_numpy(q_matrix_d).long().to(device)
        
        self.aubokt_net = AuBoKTNet(n_e, n_k,n_l ,n_sid,n_it,n_pid_try,n_stamp_try,d_e, d_k, d_l, d_a, d_x, d_p, d_it,d_try,q_matrix, q_matrix_i, q_matrix_d, u1, sigma1, h1, u2, sigma2, h2,dropout = dropout).to(device)
        self.batch_size = batch_size
    
    def train(self, train_data, test_data=None,*,epoch:int, lr=0.002, lr_decay_step=15, lr_decay_rate = 0.1,begin_idx=0) ->...:

        train_loss_list = []
        train_auc_list  = []
        test_loss_list  = []
        test_auc_list   = []

        criterion = nn.BCELoss(reduction='none')
        best_train_auc, best_test_auc = .0, .0
        
        optimizer = torch.optim.Adam(self.aubokt_net.parameters(), lr=lr, eps=1e-8, betas=(0.5, 0.999), weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)

        for idx in range(epoch):
            retain_h = []
            retain_lstm_h = []
            retain_lstm_c = []
            
            train_loss, train_auc, train_accuracy ,shuffled_ind= train_one_epoch(self.aubokt_net, begin_idx,retain_h,retain_lstm_h,retain_lstm_c,optimizer, criterion, self.batch_size, *train_data)

            train_loss_list.append(train_loss)
            train_auc_list.append(train_auc)
            
            print("Train  LogisticLoss: %.6f" %  train_loss)
            
            if train_auc > best_train_auc:
                best_train_auc = train_auc
                
            scheduler.step()
            
            for i in range(1,len(retain_h)):
                retain_h[0] = torch.cat((retain_h[0],retain_h[i]),dim=0)
                retain_lstm_h[0] = torch.cat((retain_lstm_h[0],retain_lstm_h[i]),dim=0)
                retain_lstm_c[0] = torch.cat((retain_lstm_c[0],retain_lstm_c[i]),dim=0)
            deliver_h = retain_h[0].to(device)
            deliver_lstm_h = retain_lstm_h[0].to(device)
            deliver_lstm_c = retain_lstm_c[0].to(device)

            if test_data is not None:
                test_loss, test_auc, test_accuracy= self.eval(begin_idx,shuffled_ind,deliver_h,deliver_lstm_h,deliver_lstm_c,test_data)

                test_loss_list.append(test_loss)
                test_auc_list.append(test_auc)
                
                print("[Epoch %d],Train auc: %.6f, Test auc: %.6f, Test acc: %.6f" % (idx ,train_auc, test_auc, test_accuracy))
                
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    self.save("aubokt.params")
   
        return best_train_auc, best_test_auc,train_loss_list, train_auc_list, test_loss_list, test_auc_list

    def eval(self, begin,shuffled_ind,deliver_h,deliver_lstm_h,deliver_lstm_c,test_data) -> ...:
        self.aubokt_net.eval()
        return test_one_epoch(self.aubokt_net, begin,deliver_h,deliver_lstm_h,deliver_lstm_c,self.batch_size, shuffled_ind,*test_data)
    
    def save(self, filepath) -> ...:
        torch.save(self.aubokt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)
    
    def load(self, filepath) -> ...:
        self.aubokt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)


# In[ ]:




