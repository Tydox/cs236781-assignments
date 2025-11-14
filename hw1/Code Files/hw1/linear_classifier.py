import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        #We need W=(D+1xC)
        self.weights = torch.normal(mean=0,std=weight_std,size=(self.n_features,self.n_classes))
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted class of the corresponding sample.
                Predictions are integers in range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """
        #X=NxD+1
        #Y^=Nx1
        #Score=NxC
        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        class_scores = x @ self.weights #(NxD+1)(D+1xC)=NxC
        _,y_pred = torch.max(class_scores,dim=1)#dim1=C dim
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """
        #y=Nx1
        #y^=Nx1
        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        acc = (y == y_pred).float().mean()
        
        #acc = (y==y_pred).float() #convert vector true\false to 1\0
        #acc = acc.sum()/acc.numel() #correct predictions / total predictions
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0
            
            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss, using the weight_decay parameter.

            # ====== YOUR CODE: ======
            #raise NotImplementedError()
            #TRAINING
            total_samples = 0
            for batch_idx, (X,y) in enumerate(dl_train):
                y_pred, class_scores = self.predict(X) #forward propagation
                loss = loss_fn(X, y, class_scores, y_pred) #compute loss
                loss_grad = loss_fn.grad() #compute gradient with respect to W matrix
                loss = loss + 0.5*weight_decay*torch.sum(self.weights ** 2) #add regulizer to loss lambda|w|^2
                self.weights = self.weights - learn_rate*(loss_grad + weight_decay*self.weights) #SGD update weights
                #print(epoch_idx)
                #if epoch_idx == 0:
                #    train_res.loss.append(loss/)
                #train_res.loss.append(train_res.loss[epoch_idx]+loss)
                #total_correct += self.evaluate_accuracy(y, y_pred)
                total_correct += (y_pred == y).sum().item()
                average_loss += loss.item()* X.shape[0]
                total_samples += X.shape[0]
                
                
            train_res.accuracy.append(total_correct / total_samples)
            train_res.loss.append(average_loss / total_samples)
            
            
            #VALIDATION
            # valid_correct = 0
            # valid_loss = 0.0
            total_samples = 0
            total_correct = 0
            average_loss = 0
            #with torch.no_grad():
            for X_val,y_val in dl_valid:
                y_pred_val, class_scores_val = self.predict(X_val)#forward propagation
                loss_val = loss_fn(X_val, y_val, class_scores_val, y_pred_val) #compute loss
                loss_val = loss_val + 0.5*weight_decay * torch.sum(self.weights ** 2)
                total_correct += (y_pred_val == y_val).sum().item()
                average_loss += loss_val.item()* X_val.shape[0]
                total_samples += X_val.shape[0]
                    
            #print(total_samples)
            valid_res.accuracy.append(total_correct /total_samples)
            valid_res.loss.append(average_loss / total_samples)
            #print(train_res)
            print(f"epoch {epoch_idx}: train acc: {train_res.accuracy[epoch_idx]} eval acc: {valid_res.accuracy[epoch_idx]} | train loss: {train_res.loss[epoch_idx]:.2f} eval loss: {valid_res.loss[epoch_idx]:.2f}")
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        #W=DxC
        C,H,W = img_shape
        #print(C,H,W)
        w_images = self.weights
        if has_bias:#skip w0=b
            w_images=w_images[1:]
        
        #assert W.shape[0] == C*H*W, f"Weight features {W.shape[0]} do not match img_shape {C*H*W}"
        w_images = torch.reshape(w_images.T,(self.n_classes,C,H,W))
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    hp['weight_std'] = 0.001 #noise
    hp['learn_rate'] = 0.01
    hp['weight_decay'] = 0.001
    
    
    # hp['weight_std'] = 0.01
    # hp['learn_rate'] = 0.01
    # hp['weight_decay'] = 0.001
    # ========================

    return hp
