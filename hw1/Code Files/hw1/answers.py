r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Q1.1 **False** - "equally useful train-test split" constraint,

* if we don't ensure we have the same number of samples per class in both subsets, then we will have an uneven datasplit and it may cause bias during learning to the class with more samples,

* additionally depending on the complexity of the classes, the more complex a class is - the more samples we will probably want to give it,

* also we usually use 70\30 test\train split, if we do for example 99\1 split or 1\99 split - we will get bad results and they wont be good splits. 

We need to ensure the ratio is good, we need to have enough data for training and learning the fine details but also enough "unseen" data so we can test and evaluate the model.

------------------------------------------------

Q1.2 **False** - we use cross-validation when we evaluate the train dataset for the best hyperparamters. 

we split the trainset into train\eval subsets -> we train the model on train subset -> check performance using val subset -> choose the hyperparameters from the best split performance -> take the entire train dataset (train+eval) and use the hyper paramters to train the model -> only now we use the testset

------------------------------------------------

Q1.3 **True** - we use the val subset to test performance of the model (same performance we will use for test)

------------------------------------------------

Q1.4 **False** - If this means during training then doing this means we are destroying our clean data.

Our data consists of $D=\{ \textbf{x}^\left( i \right) , y^\left( i \right)\}_{i=1}^N$ 

We use the labels to guide the model to converge into a minimum error between the real label 'y' and the estimated label $\widehat{y}$: loss(y,$\widehat{y}$)

Adding noise to the labels-y means we lie to the model when he is correct\incorrect, the model cannot properly learn.

We can add noise the x (features) to improve robustness.

If this means after Training then its **True**, if we have photos of dogs and cats, and we label some dogs as cats and our model still predicts dogs, we know the model is robust, the accuracy will go down - but there are other metrics (i dont know of) that will be better for this task of evaluation.

------------------------------------------------
"""

part1_q2 = r"""
**Your answer:**

**False** - we use cross-validation for hyper parameter tuning, lambda should be tuned before we do the final training and then the testing. 

We do not use the test to choose the best lambda hyper parameter.



"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
