import torch.nn as nn
import torch
from torch.nn import functional as F


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        loss = self.error_metric(prediction, positive_mask * label)

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss


class LBHingev2(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=None, threshold=None, return_per_sequence=False):
        super().__init__()

        if error_metric is None:
            if return_per_sequence:
                reduction = 'none'
            else:
                reduction = 'mean'
            error_metric = nn.MSELoss(reduction=reduction)

        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100

        self.return_per_sequence = return_per_sequence

    def forward(self, prediction, label, target_bb=None, valid_samples=None):
        assert prediction.dim() == 4 and label.dim() == 4

        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        # Mask invalid samples
        if valid_samples is not None:
            valid_samples = valid_samples.float()
            prediction = prediction * valid_samples
            label = label * valid_samples

            loss = self.error_metric(prediction, positive_mask * label)

            if self.return_per_sequence:
                loss = loss.mean((-2, -1))
            else:
                loss = loss * valid_samples.numel() / (valid_samples.sum() + 1e-12)
        else:
            loss = self.error_metric(prediction, positive_mask * label)

            if self.return_per_sequence:
                loss = loss.mean((-2, -1))

        return loss


class IsTargetCellLoss(nn.Module):
    def __init__(self, return_per_sequence=False, use_with_logits=True):
        super(IsTargetCellLoss, self).__init__()
        self.return_per_sequence = return_per_sequence
        self.use_with_logits = use_with_logits

    def forward(self, prediction, label, target_bb=None, valid_samples=None):
        score_shape = label.shape[-2:]

        prediction = prediction.view(-1, score_shape[0], score_shape[1])
        label = label.view(-1, score_shape[0], score_shape[1])

        if valid_samples is not None:
            valid_samples = valid_samples.float().view(-1)

            if self.use_with_logits:
                prediction_accuracy_persample = F.binary_cross_entropy_with_logits(prediction, label, reduction='none')
            else:
                prediction_accuracy_persample = F.binary_cross_entropy(prediction, label, reduction='none')

            prediction_accuracy = prediction_accuracy_persample.mean((-2, -1))
            prediction_accuracy = prediction_accuracy * valid_samples

            if not self.return_per_sequence:
                num_valid_samples = valid_samples.sum()
                if num_valid_samples > 0:
                    prediction_accuracy = prediction_accuracy.sum() / num_valid_samples
                else:
                    prediction_accuracy = 0.0 * prediction_accuracy.sum()
        else:
            if self.use_with_logits:
                prediction_accuracy = F.binary_cross_entropy_with_logits(prediction, label, reduction='none')
            else:
                prediction_accuracy = F.binary_cross_entropy(prediction, label, reduction='none')

            if self.return_per_sequence:
                prediction_accuracy = prediction_accuracy.mean((-2, -1))
            else:
                prediction_accuracy = prediction_accuracy.mean()

        return prediction_accuracy


class TrackingClassificationAccuracy(nn.Module):
    """ Estimates tracking accuracy by computing whether the peak of the predicted score map matches with the target
        location.
    """
    def __init__(self, threshold, neg_threshold=None):
        super(TrackingClassificationAccuracy, self).__init__()
        self.threshold = threshold

        if neg_threshold is None:
            neg_threshold = threshold
        self.neg_threshold = neg_threshold

    def forward(self, prediction, label, valid_samples=None):
        prediction_reshaped = prediction.view(-1, prediction.shape[-2] * prediction.shape[-1])
        label_reshaped = label.view(-1, label.shape[-2] * label.shape[-1])

        prediction_max_val, argmax_id = prediction_reshaped.max(dim=1)
        label_max_val, _ = label_reshaped.max(dim=1)

        label_val_at_peak = label_reshaped[torch.arange(len(argmax_id)), argmax_id]
        label_val_at_peak = torch.max(label_val_at_peak, torch.zeros_like(label_val_at_peak))

        prediction_correct = ((label_val_at_peak >= self.threshold) & (label_max_val > 0.25)) | ((label_val_at_peak < self.neg_threshold) & (label_max_val < 0.25))

        if valid_samples is not None:
            valid_samples = valid_samples.float().view(-1)

            num_valid_samples = valid_samples.sum()
            if num_valid_samples > 0:
                prediction_accuracy = (valid_samples * prediction_correct.float()).sum() / num_valid_samples
            else:
                prediction_accuracy = 1.0
        else:
            prediction_accuracy = prediction_correct.float().mean()

        return prediction_accuracy, prediction_correct.float()

class Contrast(nn.Module):
    def __init__(self,threshold,clip=None):
        super().__init__()
        self.clip = clip
        self.epsilon=1e-8
        self.temperature=0.05
        self.threshold = threshold if threshold is not None else -100

    def forward(self,prediction, label,target_bb=None):
        # 加上一个KL散度约束？
        num = prediction.shape[1]
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        distance_loss = torch.max(torch.tensor(0).cuda(), torch.max(-1 - prediction, prediction - 1)).sum()/num

        if distance_loss < 1:
            prediction_exp = torch.exp(prediction / self.temperature)
        else:
            prediction_exp = torch.exp(prediction)
        positive = (prediction_exp*positive_mask).squeeze(0).sum()
        negative = (prediction_exp*negative_mask).squeeze(0).sum()
        contract_loss = (-torch.log(positive/(negative+positive)))/num

        kl_loss = torch.nn.functional.kl_div((prediction*positive_mask)[0].flatten(1).softmax(-1).log(),label[0].flatten(1).softmax(-1),reduction='batchmean')

        loss = contract_loss+kl_loss
        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss,contract_loss,kl_loss,distance_loss+(prediction*negative_mask).clamp(0).sum()/num

    def forward2(self,prediction, label,target_bb=None):
        num = prediction.shape[1]
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        # mins = prediction.squeeze(0).flatten(1).min(dim=1)[0].view(1,-1,1,1)
        # maxs = prediction.squeeze(0).flatten(1).max(dim=1)[0].view(1,-1,1,1)
        # prediction = -1+2*(prediction-mins)/(maxs-mins)
        distance = torch.max(torch.tensor(0).cuda(), torch.max(-1 - prediction, prediction - 1))
        if distance.max()<10:
            prediction_exp = torch.exp(prediction / self.temperature)
        else:
            prediction_exp = torch.exp(prediction)
        positive = (prediction_exp*positive_mask).squeeze(0).sum()
        negative = (prediction_exp*negative_mask).squeeze(0).sum()
        loss = -torch.log(positive/(negative+positive))/num+distance.sum()/num
        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss
