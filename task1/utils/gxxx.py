# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:31:15 2019

@author: admin
"""
import csv 
import random

length = 0
rows = []
max_skill_num = 0
min_skill_num = 1
max_num_problems = 0

data = 'A2012_skill_id_'
with open('data/' +data +'test.csv', "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        #if row[-1] != '':
        rows.append(row)
        #else:
         #   rows.append(row[:-1])
with open('data/' +data +'train.csv', "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        rows.append(row)
index = 0
i = 0
print ("the number of rows is " + str(len(rows)/3))
tuple_rows = []
#turn list to tuple
while(index < len(rows)-1):
    problems_num = int(rows[index][0])
    tmp_max_skill = max(map(int, rows[index+1]))
    tmp_min_skill = min(map(int, rows[index+1]))
    length += int(rows[index][0])
    if(tmp_max_skill > max_skill_num):
        max_skill_num = tmp_max_skill
    if(tmp_min_skill < min_skill_num):
        min_skill_num = tmp_min_skill
    if(problems_num <= 2):
        index += 3
    else:
        if problems_num > max_num_problems:
            max_num_problems = problems_num
        tup = (rows[index], rows[index+1], rows[index+2])
        tuple_rows.append(tup)
        index += 3
#shuffle the tuple

random.shuffle(tuple_rows)
print('records: ', length)
print ("The number of students is ", len(tuple_rows))
print('skills: ', max_skill_num  )
print('skills: ', min_skill_num  )
print('Max steps: ', max_num_problems)
