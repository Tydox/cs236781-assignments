import abc
import torch
from matrepr import mdisplay

class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        N = x.shape[0]
        wyixi = x_scores[torch.arange(N),y] #(N rows andy selects the col ) -> (N,_)
        wyixi=wyixi.unsqueeze(1)
       # mdisplay(wyixi, floatfmt=".2f")
        M = x_scores - wyixi + self.delta
        M[torch.arange(N),y] = 0 #remove correct classes - the delta reminder in the y cols 
        
        M_max = torch.clamp(input=M,min=0) # NxC #hinge
        
        #mdisplay(M_max, floatfmt=".2f")
        loss_i = torch.sum(input=M_max,dim=1) #Nx_
        #mdisplay(loss_i, floatfmt=".2f")
        loss = torch.mean(loss_i)
        #mdisplay(loss, floatfmt=".2f")
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        self.grad_ctx = {'M':M,'x': x,'y': y}
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient. (DxN)(NxC)

        grad = None
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # x = self.grad_ctx['x']
        # M_mask = (self.grad_ctx['M'] > 0).float() #returns map of 1 or 0
        

        # sum_ones = torch.sum(input=M_mask,dim=1).unsqueeze(1) #Nx1
        # dLwy = -x*sum_ones
        
        
        # M_mask_zeros = (self.grad_ctx['M']  == 0).float()
        
        
        
        # dLwj = x.T @ M_mask #())(NxC)
        
        
        # mdisplay(x, floatfmt=".2f")
        # mdisplay(M_mask, floatfmt=".2f")
        # mdisplay(M_mask_zeros, floatfmt=".2f")

        # mdisplay(dLwy, floatfmt=".2f")
        # mdisplay(dLwj, floatfmt=".2f")
        
        x = self.grad_ctx['x'] # (NxD)
        M = self.grad_ctx['M'] # (NxC), margins from loss()
        y = self.grad_ctx['y'] # (Nx_)
        N = x.shape[0]
        
        G = (M>0).float() #NxC
        sum_ones = torch.sum(G,dim=1) #Nx_
        G[torch.arange(N),y] = -sum_ones
        grad = x.T @ G / N
        
        
        # ========================

        return grad
