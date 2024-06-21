"""
Module containing class for binary segmentation metrics
Original Author: Kota Yamaguchi (https://github.com/kyamagu)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
Code adapted from this gist: https://gist.github.com/kyamagu/73ab34cbe12f3db807a314019062ad43
"""
import numpy as np

def get_meter(meter_name):
    if meter_name == 'SexAgeMeter':
        return SexAgeMeter()
    elif meter_name == 'RegressionMeter':
        return RegressionMeter()
    elif meter_name == 'BinaryClassificationMeter':
        return BinaryClassificationMeter()
    else:
        raise ValueError(f'Unknown meter name: {meter_name}')

class SexAgeMeter():
    def __init__(self):
        self.sex_meter = BinaryClassificationMeter()
        self.age_meter = RegressionMeter_r2()

    def reset(self):
        self.sex_meter.reset()
        self.age_meter.reset()

    def get_metrics(self):
        sex_metrics = self.sex_meter.get_metrics()
        age_metrics = self.age_meter.get_metrics()

        return {**sex_metrics, **age_metrics}.copy()

    def update(self, output, target, return_metrics=True):
        # update
        sex_output, age_output = output[:, [0]], output[:, [1]]
        sex_target, age_target = target[:, [0]], target[:, [1]]

        curr_sex_metrics = self.sex_meter.update(sex_output, sex_target, return_metrics=return_metrics)
        curr_age_metrics = self.age_meter.update(age_output, age_target, return_metrics=return_metrics)

        # return metrics
        if return_metrics:
            return {**curr_sex_metrics, **curr_age_metrics}.copy()


class RegressionMeter_r2():
    def __init__(self, scale=100):
        self.reset()
        self.scale = scale

    def reset(self):
        self.n = 0
        self.accum_sum_error = 0
        self.accum_abs_error = 0
        self.accum_sqrd_error = 0
        self.outputs = []
        self.targets = []

    def update(
        self,
        output,
        target,
        return_metrics=True,
        print_samples=None
    ):
        if print_samples:
            for single_pred, single_target in zip(output[:print_samples], target[:print_samples]):
                print(f'pred: {single_pred.item():.2f}, target: {single_target.item():.2f}')
                print('\n-----\n')

        curr_n = len(output)
        # calculate accumulated errors
        error = self.scale * (output - target)
        curr_sum_error = np.sum(error)
        curr_abs_error = np.sum(np.abs(error))
        curr_sqrd_error = np.sum(np.square(error))

        self.accum_sum_error += curr_sum_error
        self.accum_abs_error += curr_abs_error
        self.accum_sqrd_error += curr_sqrd_error
        self.n += curr_n

        # store outputs and targets for R2 calculation
        self.outputs.extend(output)
        self.targets.extend(target)

        if return_metrics:
            return {
                'mer': curr_sum_error / curr_n,
                'mae': curr_abs_error / curr_n,
                'mse': curr_sqrd_error / curr_n,
                'r2': self._calculate_r2()
            }

    def _calculate_r2(self):
        if self.n == 0:
            return float('nan')
        outputs = np.array(self.outputs)
        targets = np.array(self.targets)
        mean_target = np.mean(targets)
        total_variance = np.sum((targets - mean_target) ** 2)
        residual_variance = np.sum((targets - outputs) ** 2)
        return 1 - (residual_variance / total_variance)

    def get_metrics(self):
        return {
            'mer': self.accum_sum_error / self.n,
            'mae': self.accum_abs_error / self.n,
            'mse': self.accum_sqrd_error / self.n,
            'r2': self._calculate_r2()
        }


class RegressionMeter():
    def __init__(self, scale=100):
        self.reset()
        self.scale = scale

    def reset(self):
        self.n = 0
        self.accum_sum_error  = 0
        self.accum_abs_error  = 0
        self.accum_sqrd_error = 0

    def update(
        self,
        output,
        target,
        return_metrics=True,
        print_samples=None
    ):
        if print_samples:
            for single_pred, single_target in zip(output[:print_samples], target[:print_samples]):
                print(f'pred: {single_pred.item():.2f}, target: {single_target.item():.2f}')
                print('\n-----\n')

        curr_n = len(output)
        # calculate accumulated errors
        error = self.scale * (output - target)
        curr_sum_error  = np.sum(error)
        curr_abs_error  = np.sum(np.abs(error))
        curr_sqrd_error = np.sum(np.square(error))

        self.accum_sum_error  += curr_sum_error
        self.accum_abs_error  += curr_abs_error
        self.accum_sqrd_error += curr_sqrd_error
        self.n += curr_n

        if return_metrics:
            return {
                'mer': curr_sum_error / curr_n,
                'mae': curr_abs_error / curr_n,
                'mse': curr_sqrd_error / curr_n
            }

    def get_metrics(self):
        return {
            'mer': self.accum_sum_error / self.n,
            'mae': self.accum_abs_error / self.n,
            'mse': self.accum_sqrd_error / self.n
        }


class BinaryClassificationMeter():
    """Class to compute and stores the average and current binary metrics"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.confusion = {
            'tp': 0, # true positive
            'tn': 0, # true negative
            'fp': 0, # false positive
            'fn': 0  # false negative
        }

        self.metrics = {
            'acc':  0, # accuracy
            'spec': 0, # specificity
            'pre':  0, # precision
            'rec':  0, # recall
            'f1':   0, # f1-score
            'iou':  0  # intersection over union
        }

    def get_metrics(self):
        return self.metrics.copy()

    def print_metrics(self):
        """Prints average metrics and returns a copy of their dictionary

        Returns:
            dict: a copy of the metric dictionary
        """
        print(f'Accuracy:    {self.metrics["acc"]  * 100:.2f}')
        print(f'Specificity: {self.metrics["spec"] * 100:.2f}')
        print(f'Precision:   {self.metrics["pre"]  * 100:.2f}')
        print(f'Recall:      {self.metrics["rec"]  * 100:.2f}')
        print(f'F1-score:    {self.metrics["f1"]   * 100:.2f}')
        print(f'IoU:         {self.metrics["iou"]  * 100:.2f}')

        return self.metrics.copy()

    def process_confusion(self, curr_confusion):
        def accuracy(confusion):
            numerator = (confusion['tp'] + confusion['tn'])
            denominator = (confusion['tp'] + confusion['tn'] + confusion['fp']+ confusion['fn'])
            return  numerator / denominator

        def specificity(confusion):
            return confusion['tn'] / (confusion['tn'] + confusion['fp'])

        def precision(confusion):
            return confusion['tp'] / (confusion['tp'] + confusion['fp'])

        def recall(confusion):
            return confusion['tp'] / (confusion['tp'] + confusion['fn'])

        def f1score(confusion):
            return 2.0*confusion['tp'] / (2.0*confusion['tp'] + confusion['fp'] + confusion['fn'])

        def iou(confusion):
            return confusion['tp'] / (confusion['tp'] + confusion['fp'] + confusion['fn'])

        # compute current metrics
        curr_acc  = accuracy(curr_confusion)
        curr_spec = specificity(curr_confusion)
        curr_pre  = precision(curr_confusion)
        curr_rec  = recall(curr_confusion)
        curr_f1   = f1score(curr_confusion)
        curr_iou  = iou(curr_confusion)

        # update values of tp, tn, fp, fn
        self.confusion['tp'] += curr_confusion['tp']
        self.confusion['tn'] += curr_confusion['tn']
        self.confusion['fp'] += curr_confusion['fp']
        self.confusion['fn'] += curr_confusion['fn']

        # update metrics
        self.metrics['acc']  = accuracy(self.confusion)
        self.metrics['spec'] = specificity(self.confusion)
        self.metrics['pre']  = precision(self.confusion)
        self.metrics['rec']  = recall(self.confusion)
        self.metrics['f1']   = f1score(self.confusion)
        self.metrics['iou']  = iou(self.confusion)

        return {
            'acc':  curr_acc,
            'spec': curr_spec,
            'pre':  curr_pre,
            'rec':  curr_rec,
            'f1':   curr_f1,
            'iou':  curr_iou
        }

    def update_with_confusion(self, confusion):
        return self.process_confusion(confusion)

    def update(self, output, target, return_metrics=True):
        """Computes the metrics for the current output-target pair
           and updates the historical average

        Args:
            output (np.array): binary predicted array/mask
            target (np.array): binary ground-truth array/mask

        Returns:
            dict: current metrics
        """
        if output.shape[1] == 1:
            # convert input types to integer
            pred  = (output > 0.5).astype(int)
            assert target.shape[1] == 1, target.shape
            truth = target.astype(int)
        elif output.shape[1] == 2:
            pred = np.argmax(output, axis=1).astype(int)
            assert target.shape[1] == 2, target.shape
            truth = np.argmax(target, axis=1).astype(int)

        # current values of tp, tn, fp, fn
        curr_tp = np.sum(pred * truth)
        curr_tn = np.sum((1 - pred) * (1 - truth))
        curr_fp = np.sum(pred * (1 - truth))
        curr_fn = np.sum((1 - pred) * truth)

        # create current confusion matrix
        curr_confusion = {
            'tp': curr_tp,
            'tn': curr_tn,
            'fp': curr_fp,
            'fn': curr_fn,
        }

        curr_metrics = self.process_confusion(curr_confusion)

        if return_metrics:
            # return current metrics
            return {'acc':  curr_metrics['acc'],
                    'spec': curr_metrics['spec'],
                    'pre':  curr_metrics['pre'],
                    'rec':  curr_metrics['rec'],
                    'f1':   curr_metrics['f1'],
                    'iou':  curr_metrics['iou']}
        else:
            # return current TP, TN, FP, FN
            return {'tp': curr_tp, 'tn': curr_tn, 'fp':  curr_fp, 'fn':  curr_fn}
