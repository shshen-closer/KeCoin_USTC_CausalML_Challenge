import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import sys
import time, datetime
from sklearn.model_selection import train_test_split, KFold

#df_train = pd.read_csv('train_data/train_task_1_2.csv')
pd.set_option('display.float_format',lambda x : '%.2f' % x)
np.set_printoptions(suppress=True)
kfold = KFold(n_splits=5, shuffle=False)

length = int(sys.argv[1])


all_data =  pd.read_csv('../Task_3_dataset/checkins_lessons_checkouts_training.csv', encoding = "ISO-8859-1", low_memory=False)

print(all_data.head())
all_data['timestamp'] =  all_data['Timestamp'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S')))



order = ['UserId','QuestionId','IsCorrect','ConstructId', 'Type','timestamp']

all_data = all_data[order]
print(all_data.head())

users = list(set(np.array(all_data['UserId'])))


c2q = {}
for one in np.array(all_data):
    ccc = int(one[3])
    if ccc in c2q.keys():
        if one[1] not in c2q[ccc]:
            c2q[ccc].append(one[1])
    else:
        c2q[ccc] = [one[1]]


with open('data/user2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        user2id = eval(line)
with open('data/problem2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        problem2id = eval(line)
with open('data/skill2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        skill2id = eval(line)

constructs = []
with open('../Task_3_dataset/constructs_input_test.csv', 'r') as fi:
    for line in fi:
        if line[0]!='C':
            constructs.append(int(line.strip()))
student_r = []
for uu in tqdm(users):
    idx = all_data[(all_data.UserId==uu)].index.tolist()
    temp1 = all_data.iloc[idx]
    temp1 = temp1.sort_values(by=['timestamp']) 
    temp = np.array(temp1)
    if len(temp) < 2:
        continue
 #   while len(temp) >= 2:
    quiz = temp[-length:]

    train_q = []
    train_a = []
    train_skill = []
    for one in range(len(quiz)):
        
        if quiz[one][4] == 'Lesson':
            train_a.append(1)
        else:
            train_a.append(int(quiz[one][2]))
        train_q.append(problem2id[quiz[one][1]])
        train_skill.append(skill2id[quiz[one][3]])
    student_r.append([train_q, train_a, train_skill])
  #  temp = temp[length:]


control = []
treat = []
jiedian =[]
jiedian_c = []

outputs = []
for i in range(len(constructs)):
    #print(i)
    item = []
    for j in range(len(constructs)):
        if i == j:
            continue
        s1 = skill2id[constructs[i]]
        s2 = skill2id[constructs[j]]

        counts = 0
        counts_c = 0
        candidate = c2q[constructs[j]]
        t_temp = []
        c_temp = []
        np.random.shuffle(student_r)
        for rec in student_r:
            train_q = rec[0]
            train_a = rec[1]
            train_skill = rec[2]
            q_o = problem2id[candidate[0]]
            if s1 in train_skill:
                temp_q = train_q  +[q_o]
                train_skill = train_skill  +[s2]
                train_a = train_a + [0]
                t_temp.append([temp_q, train_a, train_skill, len(train_a)])
                counts += 1

                train_q2 = []
                train_a2 = []
                train_skill2 = []
                for tq, ts, ta in zip(temp_q, train_skill, train_a):
                    if ts != s1:
                        train_q2.append(tq)
                        train_a2.append(ta)
                        train_skill2.append(ts)
                    else:
                        train_q2.append(q_o)
                        train_a2.append(ta)
                        train_skill2.append(s2)

                c_temp.append([train_q2, train_a2, train_skill2, len(train_a2)])
                counts_c += 1
            if len(t_temp) == 50:
                break
        treat.extend(t_temp)
        control.extend(c_temp)

            
        jiedian.append(counts)
        jiedian_c.append(counts_c)
        if counts == 0:
            print('error_t')
        if counts_c == 0:
            print('error_c')
#print(jiedian)
#print(jiedian_c)
print(len(jiedian))
print(len(jiedian_c))
print(len(treat))
print(len(control))

np.save("data/treat.npy",np.array(treat))
np.save("data/control.npy",np.array(control))
np.save("data/jiedian.npy",np.array(jiedian))
np.save("data/jiedian_c.npy",np.array(jiedian_c))

