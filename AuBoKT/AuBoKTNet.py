
import torch
from torch import nn
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AuBoKTNet(nn.Module):
    def __init__(self, n_e, n_k, n_l, n_sid, n_it, n_pid_try, n_stamp_try, d_e, d_k, d_l, d_a, d_x, d_p, d_it, d_try, q_matrix, q_matrix_i, q_matrix_d, u1, sigma1, h1, u2, sigma2, h2, dropout = 0.2):
        super(AuBoKTNet, self).__init__()
        self.d_e = d_e  
        self.d_k = d_k  
        self.d_l = d_l  
        self.d_a = d_a  
        self.n_k = n_k 
        self.d_p = d_p 
        self.d_it = d_it 
        self.d_try = d_try
        self.n_e = n_e
        self.n_k = n_k
        
        self.d_x = d_x
        self.q_matrix   = q_matrix
        self.q_matrix_i = q_matrix_i
        self.q_matrix_d = q_matrix_d

        self.lstm_cell = torch.nn.LSTMCell(input_size =d_x+d_a+d_try+d_try,hidden_size=d_x)
        
        self.e_embed  = nn.Embedding(n_e+1, self.d_e)         
        torch.nn.init.xavier_uniform_(self.e_embed.weight)
        self.k_embed  = nn.Embedding(n_k+1, self.d_k, padding_idx=0)         
        torch.nn.init.xavier_uniform_(self.k_embed.weight) 
        self.ed_embed = nn.Embedding(n_l+1, self.d_l)          
        torch.nn.init.xavier_uniform_(self.ed_embed.weight)
        self.kd_embed = nn.Embedding(n_l+1, self.d_l, padding_idx=0)          
        torch.nn.init.xavier_uniform_(self.kd_embed.weight)    
        self.a_embed  = nn.Embedding(2, self.d_a)             
        torch.nn.init.xavier_uniform_(self.a_embed.weight)
        self.ability_embed = nn.Embedding(n_sid+2, self.d_p)   
        torch.nn.init.xavier_uniform_(self.ability_embed.weight)
        self.itime_embed = nn.Embedding(n_it+1, d_it)         
        torch.nn.init.xavier_uniform_(self.itime_embed.weight)
        self.pid_try_embed = nn.Embedding(n_pid_try+1, d_try)        
        torch.nn.init.xavier_uniform_(self.pid_try_embed.weight)
        self.stamp_try_embed = nn.Embedding(n_stamp_try+1, d_try)        
        torch.nn.init.xavier_uniform_(self.stamp_try_embed.weight)

        self.linear_i = nn.Linear(d_e+d_k+d_l+d_l, d_x) 
        torch.nn.init.xavier_uniform_(self.linear_i.weight)
        
        self.linear_fratio = nn.Linear(d_k, 1)
        torch.nn.init.xavier_uniform_(self.linear_fratio.weight) 
        
        self.linear_forgetting = nn.Linear(d_x+d_it+n_k+1, n_k+1)
        torch.nn.init.xavier_uniform_(self.linear_forgetting.weight) 
        
        self.linear_pre_predicting = nn.Linear(d_try+d_x+d_x+d_x+d_p+d_try, d_x)
        torch.nn.init.kaiming_uniform_(self.linear_pre_predicting.weight) 
        
        self.linear_predicting = nn.Linear(d_x, 1)
        torch.nn.init.xavier_uniform_(self.linear_predicting.weight) 
        
        self.linear_learning = nn.Linear(d_try+d_try+d_x+d_x +d_x+d_a, 1)
        torch.nn.init.xavier_uniform_(self.linear_learning.weight)
        

        
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.soft = nn.Softmax(dim=2)

        learn_gauss = True      
        
        self.center_lc = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=learn_gauss)
        self.center_lc.data.fill_(u1)
        self.variance_lc = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=learn_gauss)
        self.variance_lc.data.fill_(sigma1)
        self.lc_height =torch.nn.Parameter(torch.FloatTensor(1), requires_grad=learn_gauss)
        self.lc_height.data.fill_(h1)

        self.center_fc = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=learn_gauss)
        self.center_fc.data.fill_(u2)
        self.variance_fc = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=learn_gauss)
        self.variance_fc.data.fill_(sigma2)
        self.fc_height =torch.nn.Parameter(torch.FloatTensor(1), requires_grad=learn_gauss)
        self.fc_height.data.fill_(h2)
        
    def forward(self, e_data, ed_data, a_data, sid_data, it_data,pid_try_data,stamp_try_data,deliver_h=None, deliver_lstm_h=None,deliver_lstm_c=None,train_epoch = False,deliver_h_list=[],deliver_lstm_h_list=[],deliver_lstm_c_list=[],begin_idx = 0):
        batch_size, seq_len = e_data.size(0), e_data.size(1)
        
        e_embed_data  = self.e_embed(e_data) #(batch_size, seq_len, d_e)  
        ed_embed_data = self.ed_embed(ed_data)#(batch_size, seq_len, d_l) 
        a_embed_data  = self.a_embed(a_data)  #(batch_size, seq_len, d_a) 
        ability_embed_data = self.ability_embed(sid_data)  #(batch_size, seq_len, d_p)
        
        to_mean = (self.q_matrix_i != 0).sum(1).view(-1,1)
        to_mean[0] = 1
        embed_q_matrix_i = self.k_embed(self.q_matrix_i)
        embed_q_matrix_d = self.kd_embed(self.q_matrix_d)
        exercises = torch.arange(self.n_e+1).reshape(-1,1).to(device)
        exercises_embedding = self.e_embed(exercises).repeat(1,self.n_k+1,1)

        scores = (exercises_embedding * embed_q_matrix_i).sum(2)/(self.d_e ** 1/2)
        scores = scores.masked_fill(self.q_matrix_i==0,-1e23)
        soft_scores = nn.Softmax()(scores)
        
        forgetting_kc = (torch.arange(self.n_k+1)).repeat(batch_size,1).to(device)
        forgetting_kc_data = self.k_embed(forgetting_kc)
        forgetting_kc_ratio = self.sig(self.linear_fratio(forgetting_kc_data)).reshape(batch_size, self.n_k+1)
     
        k_embed_data  = ((embed_q_matrix_i*((soft_scores.unsqueeze(2)).repeat(1,1,self.d_k))).sum(1))[e_data]
        kd_embed_data  = ((embed_q_matrix_d*((soft_scores.unsqueeze(2)).repeat(1,1,self.d_l))).sum(1))[e_data]
        it_embed_data = self.itime_embed(it_data)
        pid_try_embed_data = self.pid_try_embed(pid_try_data)
        stamp_try_embed_data = self.stamp_try_embed(stamp_try_data)
        
        retain_h = torch.zeros((batch_size,self.n_k+1)).to(device)
        retain_lstm_h = torch.zeros((batch_size,self.d_x)).to(device)
        retain_lstm_c = torch.zeros((batch_size,self.d_x)).to(device)
        
        if  train_epoch: 
            h_pre =  torch.zeros((batch_size,self.n_k+1)).to(device)
            h_pre[:,:]=0.4
            h_lstm_0 =  nn.init.uniform_(torch.zeros((batch_size,self.d_x))).to(device)
            c_lstm_0 =  nn.init.uniform_(torch.zeros((batch_size,self.d_x))).to(device)
        else: 
            h_pre = deliver_h
            h_lstm_0 = deliver_lstm_h
            c_lstm_0 = deliver_lstm_c

        all_learning = self.linear_i(torch.cat((e_embed_data, k_embed_data, ed_embed_data, kd_embed_data),2))

        search_idx =[i for i in range(batch_size)]
        copy_idx =[i for i in range(batch_size)]

        if train_epoch :
            begin = 0
        else :
            begin = begin_idx  
    
        pred = torch.zeros(batch_size, seq_len).to(device)    

        for t in range(begin, seq_len):
            it = it_embed_data[:, t]
            h_lstm_1,c_lstm_1 = self.lstm_cell(torch.cat((all_learning[:,t],a_embed_data[:, t],
pid_try_embed_data[:, t],stamp_try_embed_data[:, t]),1),(h_lstm_0,c_lstm_0))
            a = a_embed_data[:, t]#(batch_size, d_a)
            ability = ability_embed_data[:, t]
            e = e_data[:, t]
            
            with torch.no_grad():
                mysoft = soft_scores.clone()
            q_e = mysoft[e].view(batch_size, -1)   
            
            #forgetting
            forgetting_gate = self.linear_forgetting(torch.cat((h_lstm_0,it,h_pre),1))#[batch_size, n_k]
            forgetting_gate = self.sig(forgetting_gate)
            fc = self.fc_height*torch.exp(-0.5*(((h_pre-self.center_fc)**2)/(self.variance_fc**2)))
            h_pre_f =   h_pre-h_pre*fc*forgetting_gate*forgetting_kc_ratio
            h_pre_truth = (q_e * h_pre_f).sum(1).reshape(-1,1)
            h_pre_truth = h_pre_truth.repeat(1,self.d_x)
            
            #prediction
            pre_pred = self.linear_pre_predicting(torch.cat((pid_try_embed_data[:, t],stamp_try_embed_data[:, t],h_lstm_0,all_learning[:,t],h_pre_truth,ability),1))
            pre_pred = self.dropout(self.relu(pre_pred))
            final_pred = self.linear_predicting(pre_pred)
            final_pred = self.sig(final_pred)
            pred[:, t] = final_pred.view(-1)
            
            #learning
            learning_gate = self.linear_learning(torch.cat((pid_try_embed_data[:, t],stamp_try_embed_data[:, t],h_lstm_0,h_pre_truth,all_learning[:,t],a),1))#[batch_size, n_k+1]
            learning_gate = self.sig(learning_gate).reshape(-1,1)
            LG_T = q_e * learning_gate
            lc = self.lc_height*torch.exp(-0.5*(((h_pre_f-self.center_lc)**2)/(self.variance_lc**2)))
            h = lc*LG_T + h_pre_f #[batch_size, n_k+1]

            for i in search_idx :
                if  t==seq_len-1 or e_data[i,t+1]==0:                    
                    retain_h[i]=h[i]
                    retain_lstm_h[i] = h_lstm_1[i]
                    retain_lstm_c[i] = c_lstm_1[i]
                    copy_idx.remove(i)
            search_idx = copy.deepcopy(copy_idx)

            h_pre = h 
            h_lstm_0 = h_lstm_1
            c_lstm_0 = c_lstm_1
            
            if train_epoch and t==seq_len-1:
                deliver_h_list.append(retain_h)
                deliver_lstm_h_list.append(retain_lstm_h)
                deliver_lstm_c_list.append(retain_lstm_c)
        return pred

