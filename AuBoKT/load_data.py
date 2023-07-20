
import numpy as np
import math

class DATA(object):
    def __init__(self, max_seqlen, separate_char):
        self.separate_char = separate_char
        self.max_seqlen = max_seqlen
    '''
    data format:
    length
    problem sequence
    historical total attempt time sequence
    problem difficulty sequence
    continuous attempt time
    answer sequence
    student id sequence
    interval time sequence
    '''
    
    def load_data(self, path):
        f_data  = open(path, 'r')
        
        p_data  = []
        ptry_data  = []
        pd_data = []
        ttry_data = []
        a_data  = []
        sid_data = []
        it_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 8 !=0:
                line_data = line.split(self.separate_char)
                if len(line_data[len(line_data) - 1]) == 0: 
                    line_data = line_data[:-1]
            
            if lineID % 8 == 1:
                P = line_data
            elif lineID % 8 ==2:
                PTRY = line_data
            elif lineID % 8 ==3:
                PD = line_data
            elif lineID % 8 ==4:
                TTRY = line_data
            elif lineID % 8 ==5:
                A  = line_data
            elif lineID % 8 ==6:
                SID  = line_data
            elif lineID % 8 ==7:
                IT  = line_data
                
                P  = list(map(int, P))
                K  = list(map(int, PTRY))
                PD = list(map(int, PD))
                KD = list(map(int, TTRY))
                A  = list(map(int, A))
                SID = list(map(int, SID))
                IT = list(map(int, IT))

                
                p_data.append(P)
                ptry_data.append(PTRY)
                pd_data.append(PD)
                ttry_data.append(TTRY)
                a_data.append(A)
                sid_data.append(SID)
                it_data.append(IT)
                
        f_data.close()
        
        P_dataArray = np.zeros((len(p_data), self.max_seqlen)) #设一个矩阵，序列数量行，序列最长长度列 
        for j in range(len(p_data)):
            dat = p_data[j]
            P_dataArray[j,:len(dat)] = dat
        
        PTRY_dataArray = np.zeros((len(ptry_data), self.max_seqlen)) #设一个矩阵，序列数量行，序列最长长度列 
        for j in range(len(ptry_data)):
            dat = ptry_data[j]
            PTRY_dataArray[j,:len(dat)] = dat
            
        PD_dataArray = np.zeros((len(pd_data), self.max_seqlen)) #设一个矩阵，序列数量行，序列最长长度列 
        for j in range(len(pd_data)):
            dat = pd_data[j]
            PD_dataArray[j,:len(dat)] = dat
        
        TTRY_dataArray = np.zeros((len(ttry_data), self.max_seqlen)) #设一个矩阵，序列数量行，序列最长长度列 
        for j in range(len(ttry_data)):
            dat = ttry_data[j]
            TTRY_dataArray[j,:len(dat)] = dat
            
        A_dataArray = np.zeros((len(a_data), self.max_seqlen)) #设一个矩阵，序列数量行，序列最长长度列 
        for j in range(len(a_data)):
            dat = a_data[j]
            A_dataArray[j,:len(dat)] = dat
        
        SID_dataArray = np.zeros((len(sid_data), self.max_seqlen)) #设一个矩阵，序列数量行，序列最长长度列 
        for j in range(len(sid_data)):
            dat = sid_data[j]
            SID_dataArray[j,:len(dat)] = dat
            
        IT_dataArray = np.zeros((len(it_data), self.max_seqlen)) #设一个矩阵，序列数量行，序列最长长度列 
        for j in range(len(it_data)):
            dat = it_data[j]
            IT_dataArray[j,:len(dat)] = dat
        return P_dataArray, PD_dataArray, A_dataArray, SID_dataArray, IT_dataArray, PTRY_dataArray,TTRY_dataArray

