# -*- coding: utf-8 -*-
"""
@author: pijnenburgmgf

This is a script file for generating the 4 event logs described 
in section 4 of the paper. The logs are created using the process model
of the returned devices of the introduction section.

The programming could be more efficient, but I choose to program in such a way
that the structure is simple and easy adjustable in various ways.

The program will run fast anyway, even for large event logs.
"""

import numpy as np
import random as ran # for generating random numbers
import pandas as pd # for easily reading csv files

#If needed create the /logs directory manualy!
path = './logs/'



# define the behavior of the XOR-junctions
def junc_XOR1_beh1(PI, X):
    '''
    X  - a three dimensional list containing the feature values of the 
         case: X[1]=price, X[2] = type ('P', 'L'), X[3] = random
    PI - A Proces Instance. This is a list of previous activities
    Output is the event list with one activity added.
    '''
    XOR1ran = ran.random()
    if X[0] < 380 and X[1] == 'L':
        if XOR1ran < 0.2:
            PI.extend(['V3','Z'])
        else:
            if XOR1ran < 0.9:
                PI.extend(['V4','Z'])
            else:
                PI.extend(['V5','V6','XOR2'])
    if X[0] >= 380 and X[1] == 'L':
        if XOR1ran < 0.1:
            PI.extend(['V3','Z'])
        else:
            if XOR1ran < 0.4:
                PI.extend(['V4','Z'])
            else:
                PI.extend(['V5','V6','XOR2'])
    if X[0] < 250 and X[1] == 'P':
        if XOR1ran < 0.3:
            PI.extend(['V3','Z'])
        else:
            if XOR1ran < 0.8:
                PI.extend(['V4','Z'])
            else:
                PI.extend(['V5','V6','XOR2'])
    if X[0] >= 250 and X[1] == 'P':
        if XOR1ran < 0.2:
            PI.extend(['V3','Z'])
        else:
            if XOR1ran < 0.6:
                PI.extend(['V4','Z'])
            else:
                PI.extend(['V5','V6','XOR2'])
    return(PI)  


def junc_XOR1_beh2(PI, X):
    '''
    X  - a three dimensional list containing the feature values of the 
         case: X[1]=price, X[2] = type ('P', 'L'), X[3] = random
    PI - A Proces Instance. This is a list of previous activities
    Output is the event list with one activity added.
    '''
    XOR1ran = ran.random()
    if X[0] < 380 and X[1] == 'L':
        if XOR1ran < 0.3:
            PI.extend(['V3','Z'])
        else:
            if XOR1ran < 0.6:
                PI.extend(['V4','Z'])
            else:
                PI.extend(['V5','V6','XOR2'])
    if X[0] >= 380 and X[1] == 'L':
        if XOR1ran < 0.1:
            PI.extend(['V3','Z'])
        else:
            if XOR1ran < 0.3:
                PI.extend(['V4','Z'])
            else:
                PI.extend(['V5','V6','XOR2'])
    if X[0] < 250 and X[1] == 'P':
        if XOR1ran < 0.3:
            PI.extend(['V3','Z'])
        else:
            if XOR1ran < 0.4: # was 0.9 abusievelijk
                PI.extend(['V4','Z'])
            else:
                PI.extend(['V5','V6','XOR2'])
    if X[0] >= 250 and X[1] == 'P':
        if XOR1ran < 0.2:
            PI.extend(['V3','Z'])
        else:
            if XOR1ran < 0.4:
                PI.extend(['V4','Z'])
            else:
                PI.extend(['V5','V6','XOR2'])
    return(PI)
    
        
def junc_XOR2_beh1(PI, Y):
    '''
    Y  - the result of the test: B = bad, G = good
    PI - A Proces Instance. This is a list of previous activities
    Output is the event list with one activity added.
    '''
    XOR2ran = ran.random()
    nV5 = PI.count('XOR2')
    if Y == 'B' and nV5 > 1:
        if XOR2ran < 0.7:
            PI.extend(['V7','Z'])
        else:
            PI.extend(['V5','V6','XOR2'])
    if Y == 'G':
        PI.extend(['Z'])
    if Y == 'B' and nV5 == 1:
        if XOR2ran < 0.6:
            PI.extend(['V7','Z'])
        else:
            PI.extend(['V5','V6','XOR2'])
    return(PI)     
    
def junc_XOR2_beh2(PI, Y):
    '''
    Y  - the result of the test: B = bad, G = good
    PI - A Proces Instance. This is a list of previous activities
    Output is the event list with one activity added.
    '''
    XOR2ran = ran.random()
    nV5 = PI.count('XOR2')
    if Y == 'B' and nV5 > 1:
        if XOR2ran < 0.6:
            PI.extend(['V7','Z'])
        else:
            PI.extend(['V5','V6','XOR2'])
    if Y == 'G':
        PI.extend(['Z'])
    if Y == 'B' and nV5 == 1:
        if XOR2ran < 0.2:
            PI.extend(['V7','Z'])
        else:
            PI.extend(['V5','V6','XOR2'])
    return(PI)        

# define the behavior of the AND-junction
def junc_AND(PI, X):
    rand1 = ran.randint(0, 1)
    if rand1 == 0:
        PI.extend(['V1', 'V2', 'XOR1'])
    else:
        PI.extend(['V2', 'V1', 'XOR1'])
    return(PI)       

###################################################################
# generate distribution of initial features
# distribution 1
###################################################################


    
price = np.zeros(500)
typedef = ['P'] * 500

ran.seed(54321)
for i in range(75):
    price[i] = ran.randint(150,375)
    typedef[i] = 'L'
for i in range(75,150):
    price[i] = ran.randint(381, 3000)
    typedef[i] = 'L'
for i in range(150,400):
    price[i] = ran.randint(35, 249)
for i in range(400,500):
    price[i] = ran.randint(251, 450)
d = {'price': price, 'type': typedef}
df_distr1 = pd.DataFrame(data=d)
# randomize order
df_distr1 = df_distr1.sample(frac=1, random_state = 4290).reset_index(drop=True)

# distribution 2
price = np.zeros(500)
typedef = ['P'] * 500

ran.seed(777)
for i in range(175):
    price[i] = ran.randint(150,375)
    typedef[i] = 'L'
for i in range(175,300):
    price[i] = ran.randint(381, 3000)
    typedef[i] = 'L'
for i in range(300,400):
    price[i] = ran.randint(35, 249)
for i in range(400,500):
    price[i] = ran.randint(251, 450)
d = {'price': price, 'type': typedef}
df_distr2 = pd.DataFrame(data=d)
# randomize order
df_distr2 = df_distr2.sample(frac=1, random_state = 7941).reset_index(drop=True)



######################################################
###      CREATING THE EVENT LOGS                   ###
######################################################
        
# event_log behavior I, distribution I
ran.seed(12345)

file = open(path + 'eventlog1.txt','w')   
num = 500
for i in range(num):
    pi = ['A']
    x = df_distr1.iloc[i,:]
    file.write('%d %s %f ' % (x[0], x[1], ran.random()))
    yNG = 0
    yNB = 0
    while True:
        # read last element of process instance list and decide which junction
        # we have to go to
        if pi[-1] == 'A':
            pi = junc_AND(pi, x)
        elif pi[-1] == 'XOR1':
            pi = junc_XOR1_beh1(pi, x)
        elif pi[-1] == 'XOR2':
            ranu = ran.random()
            if ranu < 0.7:
                y = 'G'
            else:
                y = 'B'
            pi = junc_XOR2_beh1(pi, y)
        elif pi[-1] == 'Z':
            break
        else:
            print("Something went wrong.")
            break
    for s in pi:    
        file.write(s + ' ')
    file.write('\n')    
file.close() 

# event_log behavior II, distribution I
ran.seed(12346)

file = open(path + 'eventlog2.txt','w')   
num = 500
for i in range(num):
    pi = ['A']
    x = df_distr1.iloc[i,:]
    file.write('%d %s %f ' % (x[0], x[1], ran.random()))
    yNG = 0
    yNB = 0
    while True:
        # read last element of process instance list and decide which junction
        # we have to go to
        if pi[-1] == 'A':
            pi = junc_AND(pi, x)
        elif pi[-1] == 'XOR1':
            pi = junc_XOR1_beh2(pi, x)
        elif pi[-1] == 'XOR2':
            ranu = ran.random()
            if ranu < 0.7:
                y = 'G'
            else:
                y = 'B'
            pi = junc_XOR2_beh2(pi, y)
        elif pi[-1] == 'Z':
            break
        else:
            print("Something went wrong.")
            break
    for s in pi:    
        file.write(s + ' ')
    file.write('\n')
file.close() 

# event_log behavior I, distribution II
ran.seed(54321)

file = open(path + 'eventlog3.txt','w')   
num = 500
for i in range(num):
    pi = ['A']
    x = df_distr2.iloc[i,:]
    file.write('%d %s %f ' % (x[0], x[1], ran.random()))
    yNG = 0
    yNB = 0
    while True:
        # read last element of process instance list and decide which junction
        # we have to go to
        if pi[-1] == 'A':
            pi = junc_AND(pi, x)
        elif pi[-1] == 'XOR1':
            pi = junc_XOR1_beh1(pi, x)
        elif pi[-1] == 'XOR2':
            ranu = ran.random()
            if ranu < 0.7:
                y = 'G'
            else:
                y = 'B'
            pi = junc_XOR2_beh1(pi, y)
        elif pi[-1] == 'Z':
            break
        else:
            print("Something went wrong.")
            break
    for s in pi:    
        file.write(s + ' ')
    file.write('\n')
file.close() 


# event_log behavior II, distribution II
ran.seed(64321)

file = open(path + 'eventlog4.txt','w')   
num = 500
for i in range(num):
    pi = ['A']
    x = df_distr2.iloc[i,:]
    file.write('%d %s %f ' % (x[0], x[1], ran.random()))
    yNG = 0
    yNB = 0
    while True:
        # read last element of process instance list and decide which junction
        # we have to go to
        if pi[-1] == 'A':
            pi = junc_AND(pi, x)
        elif pi[-1] == 'XOR1':
            pi = junc_XOR1_beh2(pi, x)
        elif pi[-1] == 'XOR2':
            ranu = ran.random()
            if ranu < 0.7:
                y = 'G'
            else:
                y = 'B'
            pi = junc_XOR2_beh2(pi, y)
        elif pi[-1] == 'Z':
            break
        else:
            print("Something went wrong.")
            break
    for s in pi:    
        file.write(s + ' ')
    file.write('\n')   
file.close() 


