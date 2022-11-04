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
data_path = int(sys.argv[2])

    
student_data =  pd.read_csv('../Task_3_dataset/student_metadata.csv', encoding = "ISO-8859-1", low_memory=False)
students = np.array(student_data['UserId'])
print(len(students))
print(len(set(students)))
years = np.array(student_data['YearGroup'])
u2y = {}
for i, j in zip(students, years):
    u2y[i] = j
all_data =  pd.read_csv('../Task_3_dataset/checkins_lessons_checkouts_training.csv', encoding = "ISO-8859-1", low_memory=False)

print(all_data.head())
all_data['timestamp'] =  all_data['Timestamp'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S')))
all_data['Year'] =  all_data['UserId'].apply(lambda x:u2y[x])



order = ['UserId','QuestionId','IsCorrect','ConstructId', 'Type','Year','timestamp']

all_data = all_data[order]
print(all_data.head())

c2q = {}
for one in np.array(all_data):
    ccc = int(one[3])
    if ccc in c2q.keys():
        if one[1] not in c2q[ccc]:
            c2q[ccc].append(one[1])
    else:
        c2q[ccc] = [one[1]]


with open(data_path + '/user2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        user2id = eval(line)
with open(data_path + '/problem2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        problem2id = eval(line)
with open(data_path + '/skill2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        skill2id = eval(line)

task4 =  pd.read_csv('../Task_4_dataset/construct_experiments_input_test.csv', encoding = "ISO-8859-1", low_memory=False)

task4 = np.array(task4)

treat = []
control = []
jiedian = []
jiedian_c = []
for item in tqdm(task4):
    #c2q = c2q_all[item[-1]]
    idx = all_data[(all_data.Year==item[-1])].index.tolist()
    temp = all_data.iloc[idx]
    users = set(np.array(temp['UserId']))
    counts = 0
    counts_c = 0
    temp = temp.reset_index(drop=True)
    candidate2 = c2q[int(item[1])]
   # np.random.shuffle(candidate2)
    for uu in users:
        idx = temp[(temp.UserId==uu)].index.tolist()
        temp1 = temp.iloc[idx]
        temp1 = temp1.sort_values(by=['timestamp']) 
        temp1 = np.array(temp1)
        if len(temp1) < 400:
            continue

        candidate1 = c2q[int(item[2])]
        quiz = temp1[-length+len(candidate1) + 1:]

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
        st = skill2id[int(item[2])]
        sc = skill2id[int(item[1])]

        for ttt in range(len(candidate1)):
            train_skill.append(st)
            train_a.append(1)
       
        train_skill.append(sc)
        train_a.append(0)
        
        qt = []
        for ccc in candidate1:
            qt.append(problem2id[ccc])

        for pqc in candidate2[:2]:
            qc = problem2id[pqc]
            temp_q = train_q + qt +[qc]
            treat.append([temp_q, train_a, train_skill, len(train_a)])
            counts += 1

        candidate1 = c2q[int(item[3])]
      #  np.random.shuffle(candidate1)
        quiz = temp1[-length+len(candidate1) + 1:]
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
        st = skill2id[int(item[3])]
        sc = skill2id[int(item[1])]
        for ttt in range(len(candidate1)):
            train_skill.append(st)
            train_a.append(1)
        train_skill.append(sc)
        train_a.append(0)
        qt = []
        for ccc in candidate1:
            qt.append(problem2id[ccc])
        for pqc in candidate2[:2]:
            qc = problem2id[pqc]
            temp_q = train_q  + qt +[qc]
            control.append([temp_q, train_a, train_skill, len(train_a)])
            counts_c += 1
    jiedian.append(counts)
    jiedian_c.append(counts_c)
    if counts == 0:
        print('error')
print(jiedian)
print(jiedian_c)

np.save("data/treat.npy",np.array(treat))
np.save("data/control.npy",np.array(control))
np.save("data/jiedian.npy",np.array(jiedian))
np.save("data/jiedian_c.npy",np.array(jiedian_c))

