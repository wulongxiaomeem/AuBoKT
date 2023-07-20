
import logging
import numpy as np
from load_data import DATA
from AuBoKT import AuBoKT
import random
import torch
import torch.nn as nn
import os
import math
device = torch.device('cuda' if torch.cuda.is_available else 'cpu' )

batch_size = 32
n_sid = 1708            
n_e =  3162             
n_k  = 102              
n_l  = 5                
n_it = 5
n_pid_try = 90
n_stamp_try = 60

max_seqlen =1000      

d_e = 128               
d_k = 128               
d_l = 16                
d_a = 128               
d_x = 128               
d_p = 16               
d_it = 16
d_try = 16

u1 = 0
sigma1  = 0.4
h1 = 1

u2 = 0.5
sigma2 = 0.3
h2 = 0.1

dropout = 0.2

q_matrix = np.loadtxt('Q_matrix.txt')
q_matrix_idx = np.loadtxt('Q_matrix_idx.txt')
q_matrix_kcdifficulty = np.loadtxt('Q_matrix_kcdifficulty_5.txt')
q_matrix[0][0]=1
q_matrix_idx[0][0]=1 
q_matrix_kcdifficulty[0][0]=1

dat_train = DATA(max_seqlen=math.ceil(max_seqlen*0.6), separate_char=',')
dat_test  = DATA(max_seqlen=math.ceil(max_seqlen*0.2), separate_char=',')
dat_total_train = DATA(max_seqlen=math.ceil(max_seqlen*0.8), separate_char=',')
logging.getLogger().setLevel(logging.INFO)

train_data = dat_total_train.load_data('total_train_filtered_5.txt')
test_data = dat_test.load_data('test_filtered_5.txt')

new_test_data = ( np.concatenate([train_data[0], test_data[0]], axis=1),
                  np.concatenate([train_data[1], test_data[1]], axis=1),
                  np.concatenate([train_data[2], test_data[2]], axis=1),
                  np.concatenate([train_data[3], test_data[3]], axis=1),
                  np.concatenate([train_data[4], test_data[4]], axis=1),
                 np.concatenate([train_data[5], test_data[5]], axis=1),
                 np.concatenate([train_data[6], test_data[6]], axis=1)

             )
begin_idx = train_data[0].shape[1]
aubokt = AuBoKT(n_e, n_k, n_l,n_sid,n_it,n_pid_try,n_stamp_try,d_e, d_k, d_l, d_a, d_x, d_p,d_it,d_try,q_matrix=q_matrix, q_matrix_i=q_matrix_idx, q_matrix_d=q_matrix_kcdifficulty,batch_size=batch_size, u1=u1, sigma1=sigma1, h1=h1, u2=u2, sigma2=sigma2, h2=h2, dropout = dropout)
best_train_auc, best_valid_auc, train_loss_list, train_auc_list, test_loss_list, test_auc_list = aubokt.train(train_data, new_test_data, epoch=20, lr=0.004, lr_decay_step=10,begin_idx=begin_idx)

print(' best train auc %f, best test auc %f' % ( best_train_auc , best_valid_auc ))





