import os
import sys

path_name =  sys.argv[1]

list_file = os.listdir(path_name + '/')

for i in range(len(list_file)):
    if list_file[i][0] == '1':
        print(list_file[i], i)
        os.system("python test_private.py " + list_file[i] + ' ' + path_name)
