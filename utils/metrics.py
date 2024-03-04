import torch
import numpy as np
np.seterr(divide="ignore", invalid="ignore")


class Evaluator(object):
    def __init__(self, num_classes=3):
        """
            Initialize the parameters

        Args:
            num_classes (int): The number of classes
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, ) * 2)  # Confusion matrix

    def _generate_list(self):
        """
            Generate the list of TP, FP, TN, FN

        Returns:
            cm (list): The list of TP, FP, TN, FN
        """
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)  # FP = sum of each column - diagonal
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)  # FN = sum of each row - diagonal
        tp = np.diag(self.confusion_matrix)  # TP = diagonal
        tn = np.diag(self.confusion_matrix).sum() - tp  # TN = sum of diagonal - TP
        cm = [fp, fn, tp, tn]
        return cm  # Return the list of FP, FN, TP, TN

    def class_accuracy(self):
        """
            Compute the class accuracy

        Returns:
            class_accuracy (float): The class accuracy of the model
        """
        cm = self._generate_list()  # TP, FP, TN, FN
        class_accuracy = cm[2] / self.confusion_matrix.sum(axis=1)  # TP / (TP + FN + FP)
        class_accuracy = np.nanmean(class_accuracy)  # If there is nan in the list, it will be ignored
        return class_accuracy  # Return the class accuracy

    def accuracy(self):
        correct_predictions = 0
        total_predictions = 0

        for i in range(len(self.confusion_matrix)):
            correct_predictions += self.confusion_matrix[i][i]  # 对角线上的元素表示正确预测的数量
            total_predictions += sum(self.confusion_matrix[i])  # 每个类别的总预测数量

        accuracy = correct_predictions / total_predictions
        return accuracy

    def precision(self):
        """
            Compute the precision

        Returns:
            precision (float): The precision of the model
        """
        cm = self._generate_list()  # TP, FP, TN, FN
        precision = cm[2] / (cm[2] + cm[0])  # TP / (TP + FP)
        precision = np.nanmean(precision)  # If there is nan in the list, it will be ignored
        return precision  # Return the precision

    def recall(self):
        """
            Compute the recall

        Returns:
            recall (float): The recall of the model
        """
        cm = self._generate_list()  # TP, FP, TN, FN
        recall = cm[2] / (cm[2] + cm[1])  # TP / (TP + FN)
        recall_mean = np.nanmean(recall)  # If there is nan in the list, it will be ignored
        return recall.tolist(), recall_mean  # Return the recall

    def f1score(self):
        """
            Compute the F1 score

        Returns:
            F1 score (float): The F1 score of the model
        """
        cm = self._generate_list()  # TP, FP, TN, FN
        f1score = (2 * cm[2]) / (2 * cm[2] + cm[1] + cm[0])  # 2 * TP / (2 * TP + FP + FN)
        f1score = np.nanmean(f1score)  # If there is nan in the list, it will be ignored
        return f1score  # Return the F1 score

    def kappa(self):
        """
            Compute the kappa

        Returns:
            kappa (float): The kappa of the model
        """
        po_row = np.sum(self.confusion_matrix, axis=1)  # Sum of each row
        po_col = np.sum(self.confusion_matrix, axis=0)  # Sum of each column
        sum_total = np.sum(po_col)  # Sum of all elements
        pe = np.dot(po_row, po_col) / float(sum_total ** 2)  # Sum of each row * Sum of each column / Sum of all elements ^ 2
        po = np.trace(self.confusion_matrix) / float(sum_total)  # Sum of diagonal / Sum of all elements
        kappa = (po - pe) / (1 - pe)  # Kappa = (Po - Pe) / (1 - Pe)
        return kappa  # Return the kappa

    def _generate_matrix(self, predictions, targets):
        """
            Generate the confusion matrix

        Args:
            predictions (np.array): The prediction of the model
            targets (np.array): The ground truth of the model

        Returns:
            confusion_matrix (np.array): The confusion matrix of the model
        """
        mask = (targets >= 0) & (targets < self.num_classes)  # Judge whether the target is in the range of the number of classes
        label = self.num_classes * targets[mask].astype("int") + predictions[mask]  # Generate the label
        count = np.bincount(label, minlength=self.num_classes ** 2)  # Count the number of each label
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)  # Generate the confusion matrix
        return confusion_matrix  # Return the confusion matrix

    def add_batch(self, predictions, targets):
        """
            Add the batch to the confusion matrix

        Args:
            predictions (np.array): The prediction of the model
            targets (np.array): The ground truth of the model

        Returns:
            None
        """
        assert targets.shape == predictions.shape, "Please check the shape of the targets and predictions!"  # Judge whether the shape of the ground truth and the prediction is the same
        self.confusion_matrix += self._generate_matrix(predictions, targets)  # Add the confusion matrix of each batch

    def reset(self):
        """
            Reset the confusion matrix

        Returns:
            None
        """
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)  # Reset the confusion matrix


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k.mul_(1 / batch_size))
        return res

