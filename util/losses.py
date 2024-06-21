import torch
import torch.nn as nn
#from torchmetrics.functional import mean_squared_log_error

def get_loss_criterion(loss_name):
    if loss_name in ['cross_entropy', 'cel', 'CrossEntropyLoss']:
        return nn.CrossEntropyLoss()
    elif loss_name == 'bce':
        return nn.BCELoss()
    elif loss_name == 'bce_with_logits':
        return SimpleCustomSexAgeWithLogitsLoss(binary=True)
    elif loss_name == 'custom_mtl':
        return SimpleCustomSexAgeLoss()
    elif loss_name == 'custom_weighted_mtl':
        return WeightedCustomSexAgeLoss()
    elif loss_name == 'simple_weighted_loss':
        print(' Using SimpleWeightedSexAgeLoss with weights 0.02, 0.98 '.center(80, '-'))
        return SimpleWeightedSexAgeLoss()
    elif loss_name == 'mse':
        return SimpleMSELoss()
    elif loss_name == 'history_weighted_loss':
        return HistoryWeightedSexAgeLoss()
    elif loss_name == 'adaptive':
        return AdaptiveSexAgeLoss(as_log = False)
        print(" Using standard adaptive loss ".center(80, '-'))
    elif loss_name == 'log_adaptive':
        return AdaptiveSexAgeLoss(as_log = True)
        print(" Using log adaptive loss ".center(80, '-'))
    else:
        raise ValueError(f'Unknown loss name: {loss_name}')

class WeightedCustomSexAgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = torch.tensor([0.5, 0.5])
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

        self.reset()

    def reset(self):
        self.n = 0
        self.sex_loss = 0
        self.age_loss = 0

    def print_log(self, reset=True):

        ratio = self.sex_loss / self.age_loss
        # adjust the weights so their sum is one
        self.weight = torch.tensor([1/(ratio + 1), ratio / (ratio + 1)])
        total_loss = (self.age_loss + self.sex_loss)
        average_loss = total_loss / self.n

        print()
        print(f'Sex loss:     ', self.sex_loss)
        print(f'Age loss:     ', self.age_loss)
        print(f'Ratio:        ', ratio)
        print(f'----------------------------------------')
        print(f'Total loss:   ', total_loss)
        print(f'Average loss: ', average_loss)
        print()

        if reset:
            self.reset()

    def forward(self, preds, labels, **kwargs):

        # sex loss
        pred_sex, label_sex = preds[:, [0]], labels[:, [0]]
        loss1 = self.bce(pred_sex, label_sex)

        # age loss
        pred_age, label_age = preds[:, [1]], labels[:, [1]]
        loss2 = self.mse(pred_age, label_age)

        # accumulate
        batch = len(preds)
        self.sex_loss += loss1.item() * batch
        self.age_loss += loss2.item() * batch
        self.n += batch

        loss = loss1 * self.weight[0] + loss2 * self.weight[1]
        # loss = loss1 + loss2

        return loss


class SimpleWeightedSexAgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = torch.tensor([0.5, 0.5])
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

        self.reset()

    def reset(self):
        self.n = 0
        self.sex_loss = 0
        self.age_loss = 0

    def print_log(self, reset=True):

        ratio = (self.sex_loss * self.weight[0]) / (self.age_loss * self.weight[1])
        # adjust the weights so their sum is one
        total_loss = (self.age_loss + self.sex_loss)
        average_loss = total_loss / self.n

        print()
        print(f'Sex loss:     ', self.sex_loss)
        print(f'Age loss:     ', self.age_loss)
        print(f'Ratio:        ', ratio)
        print(f'----------------------------------------')
        print(f'Total loss:   ', total_loss)
        print(f'Average loss: ', average_loss)
        print()

        if reset:
            self.reset()

    def forward(self, preds, labels, **kwargs):

        # sex loss
        pred_sex, label_sex = preds[:, [0]], labels[:, [0]]
        loss1 = self.bce(pred_sex, label_sex)

        # age loss
        pred_age, label_age = preds[:, [1]], labels[:, [1]]
        loss2 = self.mse(pred_age, label_age)

        # accumulate
        batch = len(preds)
        self.sex_loss += loss1.item() * batch
        self.age_loss += loss2.item() * batch
        self.n += batch

        loss = loss1 * self.weight[0] + loss2 * self.weight[1]
        # loss = loss1 + loss2

        return loss

class SimpleMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')

        self.n = 0
        self.accumulated_loss = 0

    def reset(self):
        self.n = 0
        self.accumulated_loss = 0

    def print_log(self, reset=True):
        print(f'Loss:         ', self.accumulated_loss)
        print(f'Average loss: ', self.accumulated_loss / self.n)

        if reset:
            self.reset()

    def forward(self, y_pred, y_true, **kwargs):

        curr_loss = self.mse(y_pred, y_true)
        self.n += len(y_pred)
        self.accumulated_loss += curr_loss.item()

        return curr_loss

class SimpleCustomSexAgeLoss(nn.Module):
    def __init__(self, binary=False):
        super().__init__()

        self.reset()
        self.binary = binary
        self.bce = nn.BCELoss()

    def reset(self):
        self.n = 0
        self.sex_loss = 0
        self.age_loss = 0

    def print_log(self, reset=True):

        if self.binary:
            print(f'Sex loss:     ', self.sex_loss)
            print(f'Average loss: ', self.sex_loss / self.n)
        else:
            ratio = self.sex_loss / self.age_loss
            total_loss = (self.age_loss + self.sex_loss)
            average_loss = total_loss / self.n

            print()
            print(f'Sex loss:     ', self.sex_loss)
            print(f'Age loss:     ', self.age_loss)
            print(f'Ratio:        ', ratio)
            print(f'----------------------------------------')
            print(f'Total loss:   ', total_loss)
            print(f'Average loss: ', average_loss)
            print()

        if reset:
            self.reset()

    def forward(self, preds, labels, **kwargs):

        if self.binary:
            loss = self.bce(preds, labels)
            self.sex_loss += loss.item()
            self.n += len(preds)
            return loss

        # sex loss
        pred_sex, label_sex = preds[:, [0]], labels[:, [0]]
        loss1 = self.bce(pred_sex, label_sex)

        # age loss
        pred_age, label_age = preds[:, [1]], labels[:, [1]]
        loss2 = self.bce(pred_age, label_age)

        # accumulate
        batch = len(preds)
        self.sex_loss += loss1.item() * batch
        self.age_loss += loss2.item() * batch
        self.n += batch

        loss = loss1 + loss2

        return loss

class SimpleCustomSexAgeWithLogitsLoss(nn.Module):
    def __init__(self, binary=False):
        super().__init__()

        self.reset()
        self.binary = binary
        self.bce = nn.BCEWithLogitsLoss()

    def reset(self):
        self.n = 0
        self.sex_loss = 0
        self.age_loss = 0

    def print_log(self, reset=True):

        if self.binary:
            print(f'Sex loss:     ', self.sex_loss)
            print(f'Average loss: ', self.sex_loss / self.n)
        else:
            ratio = self.sex_loss / self.age_loss
            total_loss = (self.age_loss + self.sex_loss)
            average_loss = total_loss / self.n

            print()
            print(f'Sex loss:     ', self.sex_loss)
            print(f'Age loss:     ', self.age_loss)
            print(f'Ratio:        ', ratio)
            print(f'----------------------------------------')
            print(f'Total loss:   ', total_loss)
            print(f'Average loss: ', average_loss)
            print()

        if reset:
            self.reset()

    def forward(self, preds, labels, **kwargs):

        if self.binary:
            loss = self.bce(preds, labels)
            self.sex_loss += loss.item()
            self.n += len(preds)
            return loss

        # sex loss
        pred_sex, label_sex = preds[:, [0]], labels[:, [0]]
        loss1 = self.bce(pred_sex, label_sex)

        # age loss
        pred_age, label_age = preds[:, [1]], labels[:, [1]]
        loss2 = self.bce(pred_age, label_age)

        # accumulate
        batch = len(preds)
        self.sex_loss += loss1.item() * batch
        self.age_loss += loss2.item() * batch
        self.n += batch

        loss = loss1 + loss2

        return loss

class HistoryWeightedSexAgeLoss():
    """
    This loss averages (in each component: age and sex loss, separately) the last k + 1 losses (i.e from the current training step and the last k iterations) with weights in a geometric progression (starting at beta^0 for current loss to beta^{k} for loss k steps behind)

    Then, it implements a simple weighted average on this loss with parameter w in R^2: loss = w1 * final_age_loss + w2 * final_sex_loss

    Parameters:
        binary (bool): if true, does only binary classification (sex classification) task, without multitasking. Else, does multitask (sex classification + age regression)
        k (int): number of past epochs to consider
        beta (float in [0,1]): ratio of geometric progression
        weight (array of two floats): how to weight age and sex loss
    """
    def __init__(self, 
                 binary : bool = False, 
                 k : int = 5, 
                 beta = 0.85, 
                 weight : torch.Tensor = torch.tensor([0.02, 0.98])):
        self.binary : bool = binary
        self.k : int = k
        self.beta = beta
        self.weight = weight
        self.k_past_sex_losses : torch.Tensor = torch.zeros(k)
        self.k_past_age_losses : torch.Tensor = torch.zeros(k)
        self.n : int = 0 # number of training steps passed
        # total sex_loss since last reset
        self.running_sex_loss = 0.0
        # total age_loss since last reset
        self.running_age_loss = 0.0

        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

        self.reset()

    def reset(self):
        self.n = 0
        self.k_past_sex_losses.fill_(0.0)
        self.k_past_age_losses.fill_(0.0)
        self.running_sex_loss = 0.0
        self.running_age_loss = 0.0

    def print_log(self, reset=True):

        if self.binary:
            print(f'Sex loss:     ', self.running_sex_loss)
            print(f'Average loss: ', self.running_sex_loss / self.n)
        else:
            ratio = self.running_sex_loss / self.running_age_loss
            total_loss = (self.running_age_loss + self.running_sex_loss)
            average_loss = total_loss / self.n

            print()
            print(f'Sex loss:     ', self.running_sex_loss)
            print(f'Age loss:     ', self.running_age_loss)
            print(f'Ratio:        ', ratio)
            print(f'----------------------------------------')
            print(f'Total loss:   ', total_loss)
            print(f'Average loss: ', average_loss)
            print()

        if reset:
            self.reset()

    def forward(self, preds, labels, **kwargs):
        if self.binary:
            # no weighted with weight vector
            raw_sex_loss = self.bce(preds, labels)
            self.k_past_sex_losses.roll(-1)
            self.k_past_sex_losses[-1] = raw_sex_loss 

            loss = torch.tensor([0.0])
            for i in range(self.k):
                loss[0] += (self.beta ** (self.k - 1 - i)) * self.k_past_sex_losses[i]
            loss /= (self.beta ** (self.k + 1) - 1) / (self.beta - 1) ## geometric progression sum

            self.running_sex_loss += loss.item()
            self.n += len(preds)
            return loss

        # sex loss
        pred_sex, label_sex = preds[:, [0]], labels[:, [0]]
        raw_sex_loss = self.bce(pred_sex, label_sex)

        # age loss
        pred_age, label_age = preds[:, [1]], labels[:, [1]]
        raw_age_loss = self.mse(pred_age, label_age)

        self.k_past_sex_losses.roll(-1)
        self.k_past_sex_losses[-1] = raw_sex_loss
        self.k_past_age_losses.roll(-1)
        self.k_past_age_losses[-1] = raw_age_loss

        sex_loss = torch.tensor([0.0])
        age_loss = torch.tensor([0.0])
        for i in range(self.k):
            sex_loss[0] += (self.beta ** (self.k - 1 - i) * self.k_past_sex_losses[i])
            age_loss[0] += (self.beta ** (self.k - 1 - i) * self.k_past_age_losses[i])
        sex_loss /= (self.beta ** (self.k + 1) - 1) / (self.beta - 1) ## geometric progression sum
        age_loss /= (self.beta ** (self.k + 1) - 1) / (self.beta - 1) ## geometric progression sum

        self.running_sex_loss += sex_loss
        self.running_age_loss += age_loss
        self.n += len(preds)

        loss = self.weight[0] * sex_loss + self.weight[1] * age_loss
        return loss

class AdaptiveSexAgeLoss(nn.Module):
    def __init__(self,
            as_log: bool = False):
        super().__init__()

        self.as_log = as_log
        self.n : int = 0 # number of training steps passed
        # total sex_loss since last reset
        self.running_sex_loss = 0.0
        # total age_loss since last reset
        self.running_age_loss = 0.0
        self.running_weighted_loss = 0.0

        self.mse = nn.MSELoss(reduction='mean')
        #self.mse = mean_squared_log_error()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

        self.reset()

    def reset(self):
        self.n : int = 0 # number of training steps passed
        # total sex_loss since last reset
        self.running_sex_loss = 0.0
        # total age_loss since last reset
        self.running_age_loss = 0.0
        self.running_weighted_loss = 0.0
    
    def print_log(self, reset=True):

        ratio = self.running_sex_loss / self.running_age_loss
        average_loss = self.running_weighted_loss / self.n

        print()
        print(f'Sex loss (non-weighted):     ', self.running_sex_loss)
        print(f'Age loss (non-weighted):     ', self.running_age_loss)
        print(f'Ratio sex/age (of non-weighted losses):        ', ratio)
        print(f'----------------------------------------')
        print(f'Total loss (weighted):   ', self.running_weighted_loss)
        print(f'Average loss (weighted): ', average_loss)
        print()

        if reset:
            self.reset()

    def forward(self, preds, labels, **kwargs):

        self.sigma1 = kwargs['model'].sigma1
        self.sigma2 = kwargs['model'].sigma2

        sex_preds, age_preds = preds[:, [0]], preds[:, [1]]
        sex_labels, age_labels = labels[:, [0]], labels[:, [1]]

        sex_loss = self.bce(sex_preds, sex_labels)
        age_loss = self.mse(age_preds, age_labels)
        
        self.running_sex_loss += sex_loss
        self.running_age_loss += age_loss
        self.n += len(preds)

        ## HACK FIXME: weird ".module" hack to access parameter when using DataParallel
        ## see https://discuss.pytorch.org/t/how-to-reach-model-attributes-wrapped-by-nn-dataparallel/1373/3
        if self.as_log:
            final_loss = age_loss / (2 * torch.exp(2 * self.sigma1)) + sex_loss / (torch.exp(2 * self.sigma2)) + self.sigma1 + self.sigma2
            #final_loss = age_loss / (2 * torch.exp(2 * kwargs['model'].module.sigma1)) + sex_loss / (torch.exp(2 * kwargs['model'].module.sigma2)) + kwargs['model'].module.sigma1 + kwargs['model'].module.sigma2
            self.running_weighted_loss += final_loss
            return final_loss
        else:
            final_loss = age_loss / (2 * kwargs['model'].sigma1 ** 2) + sex_loss / (kwargs['model'].sigma2 ** 2) + torch.log(kwargs['model'].sigma1) + torch.log(kwargs['model'].sigma2) 
            self.running_weighted_loss += final_loss
            return final_loss

    def apply_epsilon_constraint(self, epsilon=1e-4):
        # Aplica uma restrição mínima para garantir que sigma1 e sigma2 não sejam menores que epsilon
        self.sigma1 = torch.max(self.sigma1, torch.tensor(epsilon))
        self.sigma2 = torch.max(self.sigma2, torch.tensor(epsilon))
