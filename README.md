# KeCoin_USTC_CausalML_Challenge

## Dependencies:

- python >= 3.7
- tesorflow-gpu >= 2.0 
- numpy
- tqdm
- utils
- pandas
- sklearn

## Usage

First, put Task_1_dataset and Task_3_dataset in same file path with Task_3_dataset and Task_4_dataset.

Solutions of task3 and task4 are based on our previous work published in KDD'21
```
@inproceedings{10.1145/3447548.3467237,
author = {Shen, Shuanghong and Liu, Qi and Chen, Enhong and Huang, Zhenya and Huang, Wei and Yin, Yu and Su, Yu and Wang, Shijin},
title = {Learning Process-Consistent Knowledge Tracing},
year = {2021},
isbn = {9781450383325},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3447548.3467237},
doi = {10.1145/3447548.3467237},
pages = {1452–1460},
numpages = {9},
location = {Virtual Event, Singapore},
series = {KDD '21}
}
```
Specifically, for task3: 
first, go to task3/:
`cd task3`

then, preparing the data:
`python data_pre.py`


`python data_save.py 200`

`python data_3.py 200`

then, training the model and making predictions:
`python train_lpkt.py 0`

`python test4task3.py {model_name}`

finally, prepare submission:
`python processing.py`
