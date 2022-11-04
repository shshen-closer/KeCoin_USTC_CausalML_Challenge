# KeCoin_USTC_CausalML_Challenge

## Dependencies:

- python >= 3.8  (or >= 3.8.0 )
- tensorflow-gpu == 2.5.0  (or >= 2.5.0 ) 
- torch==1.8.0 (or >= 1.8.0 )
- numpy
- tqdm
- utils
- pandas
- sklearn
- any other necessary package
## Hardware:
A GPU with more than 20G memory, such as NVIDIA RTX 3090 we used


## Before

First, put Task_1_dataset, Task_2_dataset and Task_3_dataset in same file path with Task_4_dataset.


## For task1, we used a supervised method based on the label in local dev dataset: 
**To save time, you can directly repeat our results:**

first, go to task1/:

`cd task1`

`mkdir results`

then, 

`python test_all_private.py runs_9_3`

`python meaning.py`

**The output adj_matirx.npy is our result**

**To train from start:**

first, go to task1/:

`cd task1`

then, 

`python train_all.py`

then, 

`python test_all_private.py runs`

`python meaning.py`

**The output adj_matirx.npy is our result**



## For task2, just run the task2_private.ipynb file: 

Our main idea is to code constructs to represent the differences among them. At the same time, in terms of specific operations, we will intercept a sequence with a length of 100 from the task1 dataset, and train the model by predicting the state of the next step. Since the definition of CATE is to calculate the difference between the two steps after applying different constructs, we use the trained model to predict twice consecutively, and then calculate the difference of the results to obtain our predicted CATE value.

The model input consists of three parts. The first part is the value of each construct in the sequence, with a total of 50 values. The second part is the ID of the imposed construct, and the last part is the value corresponding to the imposed construct.

After two times of model prediction and difference, we get the CATE value to be predicted. We use three seeds to initialize our model, so we take the average value of the output of the three models as the final result.Each dataset trains a model and predicts 10 results, a total of five data sets, so the final output is a 5 * 10 matrix




## Solutions of task3 and task4 are based on our previous work published in KDD'21 -- [Learning Process-Consistent Knowledge Tracing](https://doi.org/10.1145/3447548.3467237).

The basic causal insight in this paper is that **any time spent for learning (e.g., answering questions or taking lessons) will increase knowledge, and any time passed without learning (forgetting along with time) will decrease knowledge**. For this competition, we do not use time information.

## For task3:

first, go to task3/:

`cd task3`

then, preparing the data:

`python data_pre.py`

`python data_save.py 200`

`python data_3.py 200`

then, training the model and making predictions:

`python train_lpkt.py 0`

after training, the model file will be stored in runs/, you can find model_name (e.g., 1667485939) in there.

`python test4task3.py {model_name}` 

**It takes several hours to finish this, because we choose more than 1 million samples to output stable results as possible. **

finally, prepare submission:

`python processing.py`

**The output adj_matirx.npy is our result. Noting that it is hard to keep the same results due to we have different seeds for training and choose different samples for testing every time, but the deviation should be acceptable**

## For task4: 
first, go to task4/:

`cd task4`

then, preparing the data:

`python data_pre.py`

`python data_save.py 200`

`python data_4.py 200`

then, training the model and making predictions:

`python train_lpkt.py 0`

after training, the model file will be stored in runs/, you can find model_name (e.g., 1667485939) in there.

`python test4task4.py {model_name}`

**The output cate_estimate.npy is our result. Noting that it is hard to keep the same results due to we have different seeds for training and different records for testing every time, but the deviation should be acceptable**
