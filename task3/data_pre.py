import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import time, datetime
from sklearn.model_selection import train_test_split

#df_train = pd.read_csv('train_data/train_task_1_2.csv')
pd.set_option('display.float_format',lambda x : '%.2f' % x)
np.set_printoptions(suppress=True)


student_data =  pd.read_csv('../Task_3_dataset/student_metadata.csv', encoding = "ISO-8859-1", low_memory=False)
students = np.array(student_data['UserId'])
print(len(students))
print(len(set(students)))
years = np.array(student_data['YearGroup'])
u2y = {}
for i, j in zip(students, years):
    u2y[i] = [j]
all_data =  pd.read_csv('../Task_3_dataset/checkins_lessons_checkouts_training.csv', encoding = "ISO-8859-1", low_memory=False)


print(all_data.head())
all_data['timestamp'] =  all_data['Timestamp'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S')))
all_data['Year'] =  all_data['UserId'].apply(lambda x:u2y[x])



order = ['UserId','QuestionId','IsCorrect','ConstructId', 'Year','timestamp']
all_data = all_data[order]
print(all_data.head())

print(all_data.isnull().sum())

skill_id = np.array(all_data['ConstructId'])
skills = set(skill_id)
print('skills:',  len(skills))


user_id = np.array(all_data['UserId'])
print(1)
problem_id = np.array(all_data['QuestionId'])
print(1)

user = set(user_id)
print(2)
problem = set(problem_id)

print('lll')
print(len(user), len(problem))


user2id ={}
problem2id = {}
skill2id = {}

count = 0
for i in user:
    user2id[i] = count 
    count += 1
count = 0
for i in problem:
    problem2id[i] = count 
    count += 1

count = 0
for i in skills:
    skill2id[i] = count 
    count += 1

with open('data/user2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(user2id))
with open('data/problem2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(problem2id))
with open('data/skill2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(skill2id))
'''
it_id = []
length = []
for item in tqdm(user):

    idx = all_data[(all_data.UserId==item)].index.tolist() 
    temp1 = all_data.iloc[idx]
    temp1 = temp1.sort_values(by=['timestamp']) 
    
    temp = np.array(temp1)
    length.append(len(temp))
    if len(temp) < 2:
        continue

    for iii in range(1, len(temp)):
        a = (temp[iii][-1] - temp[iii-1][-1]) / 60
        a = int(a)
        
        if a > 43200:
            a = 43200
        it_id.append(a)

print('length:',  np.mean(length))
np.save('data/it_id', np.array(it_id))  
print('its:',  len(it_id))
it = set(it_id) 
print('its:',  len(it))
it2id = {}

count = 0
for i in it:
    it2id[i] = count 
    count += 1
with open('data/it2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(it2id))
'''
print('complete')
  



 