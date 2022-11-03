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



with open('data/user2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        user2id = eval(line)
with open('data/problem2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        problem2id = eval(line)
with open('data/skill2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        skill2id = eval(line)
user_id = np.array(all_data['UserId'])
user = list(set(user_id))
train_all_id, test_id = train_test_split(user,test_size=0.05,random_state=5)
print(test_id[:5])

count = 0
    

q_a_train = []

for item in tqdm(train_all_id):

    idx = all_data[(all_data.UserId==item)].index.tolist() 
    temp1 = all_data.iloc[idx]
    temp1 = temp1.sort_values(by=['timestamp']) 
    temp = np.array(temp1)
    if len(temp) < 2:
        continue

   # while len(temp) >= 2:
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

    q_a_train.append([train_q, train_a, train_skill, len(train_q)])
    # temp = temp[length:]

q_a_valid = []
for item in tqdm(test_id):

    idx = all_data[(all_data.UserId==item)].index.tolist() 
    temp1 = all_data.iloc[idx]
    temp1 = temp1.sort_values(by=['timestamp']) 
    temp = np.array(temp1)
    if len(temp) < 2:
        continue
    #  while len(temp) >= 2:
    quiz = temp[-length:]

    test_q = [ ]
    test_a = [ ]
    test_skill = [ ]

    for one in range(len(quiz)):

        
        if quiz[one][4] == 'Lesson':
            test_a.append(1)
        else:
            test_a.append(int(quiz[one][2]))
        test_q.append(problem2id[quiz[one][1]])
        test_skill.append(skill2id[quiz[one][3]])

    q_a_valid.append([ test_q, test_a, test_skill, len(test_q)])
    # temp = temp[length:]
np.random.seed(10)
np.random.shuffle(q_a_train)
np.random.seed(10)
np.random.shuffle(q_a_valid)
np.save("data/train" + str(count) + ".npy",np.array(q_a_train))
np.save("data/test" + str(count) + ".npy",np.array(q_a_valid))




print('complete')
            



 