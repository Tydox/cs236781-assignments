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

We do not use the testset to choose the best lambda hyper parameter. because then we bias our model to the data.



"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
Soft Margin SVM Loss Function:
$L(w)=\frac{1}{n}\sum^{i=1}_{N} L_{i}(w) + \frac{\lambda}{2}||w||^{2}$
And using the hinge loss that sums how many incorrect classes were for each class:
$L_{i}({W}) =  \sum_{j \neq y_i} \max\left(0, \Delta+ \vec{w_j} \vec{x_i} - \vec{w_{y_i}} \vec{x_i}\right),$
$\hat{y_{i}}=score\space of\space correct\space class$
$\hat{y_{j}}=score\space of\space INCORRECT \space class$
we want to get as small of an error as possible for $L(w)$ meaning we want the sum to also be small.
In order to do that we need to get negative number in the max function, i.e. $0=max(0,(\color{red}-\color{black}))$ , this gives us the condition of: $\hat{y_{j}}-\hat{y_{i}}+\Updelta<0$
then we get that: $\Updelta \geq \hat{y_{i}} - \hat{y_{j}}$ , we interoperate that as $\Updelta$ defining the minimum difference between right and wrong scores of prediction. If the difference is not bigger than $\Updelta$ then we have a penalty that we add to the loss, in return this forces the model to have bigger differences between right and wrong scores predictions which gives the model a higher confidence, it is like a margin that controls how far the correct class scores must be above incorrect class scores.

if $\Updelta < 0$ then we get  $\Updelta > \hat{y_{j}} - \hat{y_{i}}$ which means that incorrect classifications do not contribute to the loss because we get $0=max(0,(\color{red}-\color{black}))$ for every incorrect classification (in addition to correct classification), the optimization no longer pushes the correct class scores to be higher so there is no confident buildup.  taking to the extreme, if $\Updelta \rightarrow -\infty$ then $L(w)=\frac{1}{n}\sum^{i=1}_{N} L_{i}(w) + \frac{\lambda}{2}||w||^{2} \approx 0+ \frac{\lambda}{2}||w||^{2} \approx \frac{\lambda}{2}||w||^{2}$ so the optimization problem of $\min L(w) = min \frac{\lambda}{2}||w||^{2}$ which means that all hinge losses become 0 no matter the scores, all the weights go to 0 because that will be the solution for that problem.
"""

part2_q2 = r"""
**Your answer:**

The interpretation of what the linear model is learning when we reshaped them into images, clearly looks like a correlation map\filter to each number, it's easy to see on 0,2,3. So we say the model is learning a kernel (filter map) that has the highest correlation with each class (numbers 0-9).

We think the classification error is based on the fact that some numbers can look the same or have high correlation because they overlap each other (מוכלים אחד בשני), for example if you look at 7 and 1, on the 1 filter you can see the shape of 7 (Z like shape) with highest activation in the middle as a line | . 
the numbers 5 and 8 look similar, which can happen from the hand writing (italic, stroke thickness) when we look at the samples some of them are also not good data, we saw a 4 that does not look like a 4 because its missing the leg. so the data has outliers that can also skew the learning.
As we learnt in the lecture about the short coming of a linear classifier, like the the horse classification that points to 2 directions.


"""

part2_q3 = r"""
**Your answer:**
1. based on the graph of the training set loss,  we see the learning rate is good, because the loss function decreases quickly and smoothly (no spikes that can warrant התבדרות of a local minimum (going backwards)).
When we look at the other 2 cases, one where the learning rate is very small and one were its big.

When Learning Rate is small:
we will get a very slow decreasing loss function, the learning is very slow because each iteration we barely move down the gradient step, this means that we need many more epochs to achieve good results - if we can even get good results because when the learning rate is very small, we can get stuck in a local minimum and without a boost of the learning rate we wont be able to escape it.
We set lr=1e-5 and saw that the loss function is linear and slowly decreasing and while the accuracy went up, it seems to start converging around 70% accuracy, meaning with more epochs we assume that we will get stuck in high local minimum.

When Learning Rate is large:
we will get a loss function that decreases but also has peaks of exploding gradient of sort, and there is no grantee of good convergence, it all depends on the data complexity and noise. The graph will be zigzagy and unstable and wont converge well. it will overshoot the local minimum.
The accuracy will also suffer big hits when the loss has peaks.


2. based on the graph of the training\test accuracy, we say the model is slightly overfitted to the training set, we can see that around epoch 5, the accuracy performance between the train\test begins to become wider, though both are growing up, but when we see a widening accuracy\loss between both sets, it's a sign for a possible overfit.


"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

the ideal pattern in 

in linear regression model we have $y=wx+b+e$

the theoretical `e` error term are assumed to be: 
1. normally distributed
2. homoscedastic - the same variance at every X
3. independent IID
If these assumtions are true, then the observed residuals error: $e_{i}=y_{i}-\hat{y_{i}}$ should behave in a similar fashion.

the ideal pattered in residual pot should have the follwing:
1. residuals errors randomly scattered around 0 (mean)
2. constant spread showing homoscedastic - if the margin size changes then the model predictions are less accurate at certain ranges and we need to think about possible transformation to make the data sample space be linearly represented.
3. no cruvature - we want a straight line - if there is a curved shape the model is missing a non linear relationship  and we need to add polynomial features or a nonlinear model.
4. no clustring aka grouping - systematically biased.

if these are valid it means the model was able to capture a systematic linear structure and the remining error is mainly noise.

Judging our residual plot, we can see that:
1. the residual are roughly centered around zero - $\checkmark$
2. the variance slighty increases between $\hat{y} [30-50]$ range, so this means that homoscedastic is mild, the model will be less consistant at higher target values.
3. no obvious curvature - the model isnt obiously missing a major nonlinear relationship.- $\checkmark$
4. outliers - some points go above $\pm 15$ 
5. both train\test seem to be in the same range - no overfit is visible - the model generalized well.

Overall we got $R^{2} >0.85$ which is high, the model fits well but not perfectly.

LSTAT - $R^{2}=0.54$ okish linear relationship
RM - $R^{2}=0.48$ okish linear relationship
PTRATIO - $R^{2}=0.26$ bad linear relationship, has clustering $\times$
INDUS - $R^{2}=0.23$ bad linear relationship
TAX - $R^{2}=0.22$ bad linear relationship, has clustering $\times$
All 5 of these are noisy and show a curvature (heteroscedacity) clearly visible in LSTAT and INDUS.
Non of these features can alone capture and describe the MEDV well,
"""

part3_q2 = r"""
**Your answer:**

**Is this still a linear regression model? Why or why not?**
The model is still linear in the parameters not necessarily linear in the features
we found a linear model after we augmented the features to be represented in a linearly separable higher dimension We map the data into a higher-dimensional feature space where nonlinear patterns become linearly separable.

**Can we fit any non-linear function of the original features with this approach?**
As we learnt in the lecture, we can theoretically approx any smooth function using the universal approx theorm. but in practice this is not feasible or good, we have the curse of dimensionality and overfitting problem and more.  

**Imagine a linear classification model. As we saw in Part 2, the parameters W of such a model define a hyperplane representing the decision boundary. How would adding non-linear features affect the decision boundary of such a classifier? Would it still be a hyperplane? Why or why not?**

For a linear classifier, the decision boundary is still a hyperplane in the transformed space $\phi(x)$, but when mapped back to the original input space it becomes nonlinear. $y=w^{T}\phi(x)$ is linear but $\phi(x)$ is not linear in $x$.

"""

part3_q3 = r"""
**Your answer:**


We will use the law of total expectation to calculate the joint expectation:
$E_{x,y}[f(x,y)]=E_x[g(x)]=E_x[E_{y|x}[f(x,y)]]=\overbrace{E_x[\underbrace{E_{y|x}[y=x]}_{\text{Sum Over Y}}]}^{\text{sum over X}}$
We will do the expectation by parts, first the inner and then the outer part.

$E_{y|x}[y|x]=\int^{1}_{0}|y-x|dy\overset{seperate \space 2 \space cases}{=}\underbrace{-\int^{x}_{0}(y-x)dy}_{\text{Y<X so |y-x|=-(y-x)}}+\int^{1}_{x}(y-x)dy=(xy-\frac{y^2}{2})\bigg\rvert_{0}^{x}+(\frac{y^2}{2}-xy)\bigg\rvert_{x}^{1}=x^{2}-x+\frac{1}{2}$
$E_x[g(x)]=\int^{1}_{0}(x^{2-x+0.5)dx}= (\frac{x^3}{3} -x+\frac{1}{2}  )\bigg\rvert_{0}^{1}=\frac{1}{3}-0.5+0.5=\boxed{\frac{1}{3}=E_{x,y}[f(x,y)]}$

For $E_x[|\hat{x}-x|]$ we will do the same as above, but we must remember that $\hat{x}$ is actually a generated number (const).
$E_{x|\hat{x}}[x|\hat{x}]=\int^{1}_{0}|\hat{x}-x|dx\overset{seperate \space 2 \space cases}{=}\int^{\hat{x}}_{0}(\hat{x}-x)dx-\int^{1}_{\hat{x}}(\hat{x}-x)dx=(x\hat{x}-\frac{x^2}{2})\bigg\rvert_{0}^{\hat{x}}-(\frac{x^2}{2}-x\hat{x})\bigg\rvert_{\hat{x}}^{1}=\boxed{x^{2}-x+\frac{1}{2}=E_x[|\hat{x}-x|]}$


We can drop the value of the scalar of the polinom because when we do derivative it will be 0 and won't contribute to the solution.
$argmin\space L_{ED}(\theta)=argmin\space 2E[|\hat{x}-x|]=argmin\space 2(x^{2}-x+0.5)==argmin\space 2(x^{2}-x)$ because $\dfrac{\partial L_{ED}(\theta)}{\partial d} = \dfrac{\partial 2(x^{2}-x+0.5)}{\partial \theta}= \dfrac{\partial 2(x^{2}-x)}{\partial \theta}=2(2x-1)=4x-2$

We got the save derivative, meaning that adding constants do not change the minimizer, only global scaling will (the 2 in the begining)

"""

# ==============

# ==============
