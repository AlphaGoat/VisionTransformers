"""
Module for loss functions.

Author: Peter Thomas
Date: 07 October 2025
"""
import torch
from torchvision import ops
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


class DETRLoss(torch.nn.Module):
    def __init__(self, num_classes, class_weight=1.0, giou_weight=1.0, bbox_weight=1.0):
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.giou_weight = giou_weight
        self.bbox_weight = bbox_weight

    @torch.no_grad()
    def hungarian_matcher(self, preds, targets):
        pred_bboxes = preds["pred_boxes"]
        pred_probs = preds["pred_logits"]
        batch_size, num_queries = pred_bboxes.size(0), pred_bboxes.size(1)

        # Flatten batch and query dimension of predictions
        pred_bboxes = pred_bboxes.flatten(0, 1) # [batch_size * num_queries, 4]
        pred_probs = pred_probs.flatten(0, 1) # [batch_size * num_queries, 4]

        # Flatten target / num boxes dimensions for target bboxes
        target_bboxes = torch.cat([t["boxes"] for t in targets])
        target_classes = torch.cat([t["labels"] for t in targets])

        # Class cost is negative of pred probability given for target class
        C_classes = -1. * pred_probs[:, target_classes]

        # L1 norm for bounding box cost
        C_boxes = torch.cdist(pred_bboxes, target_bboxes, p=1)

        C_giou = -ops.generalized_box_iou(
            ops.box_convert(pred_bboxes, in_fmt='cxcywh', out_fmt='xyxy'),
            ops.box_convert(target_bboxes, in_fmt='cxcywh', out_fmt='xyxy'),
        )

        C_total = self.class_weight * C_classes + \
            self.bbox_weight * C_boxes + \
            self.giou_weight * C_giou
        C_total = C_total.view(batch_size, num_queries, -1)
        C_total = C_total.cpu().detach().numpy()

        # Find the optimum pairs that produce the minimum summation
        indices = [linear_sum_assignment(C_total[i]) for i in range(batch_size)]

        return [(torch.IntTensor(i), torch.IntTensor(j)) for i, j in indices]

    def loss_bboxes(self, p_bboxes, t_bboxes, n_boxes):
        # Bbox loss is just L1 loss
        loss_bbox = F.l1_loss(p_bboxes, t_bboxes, reduction="none")

        # Normalize loss by number of boxes in each batch
        loss_bbox /= n_boxes
        return loss_bbox

    def loss_giou(self, p_bboxes, t_bboxes, n_boxes):
        loss_giou = 1. - torch.diag(
            ops.generalized_box_iou(
                ops.box_convert(p_bboxes, in_fmt='cxcywh', out_fmt='xyxy'),
                ops.box_convert(t_bboxes, in_fmt='cxcywh', out_fmt='xyxy'),
            )
        )
        loss_giou /= n_boxes
        return loss_giou

    def loss_class(self, p_probs, t_classes, n_boxes, pad_mask):
        loss_class = torch.nn.CrossEntropyLoss()(p_probs, t_classes, reduction="none")
        loss_class *= pad_mask
        loss_class /= n_boxes
        return loss_class

    def __call__(self, preds, targets):

        # Hungarian matching to find optimum pairs
        pred_idxs, target_idxs = self.hungarian_matcher(preds, targets)
        pred_idxs, target_idxs = torch.IntTensor(pred_idxs), torch.IntTensor(target_idxs)

        # Order target indices (makes tracking a little easier...)
        pred_idxs = pred_idxs[target_idxs.argsort()]

        # Unpack predictions and targets
        pred_bboxes = preds["pred_boxes"]
        target_bboxes = targets["boxes"]

        pred_probs = preds["pred_logits"]
        target_classes = targets["labels"]

        # Zero out predictions corresponding to a pad target box
        pad_mask = targets["pad_mask"]
        pred_bboxes *= pad_mask
        pred_probs *= pad_mask

        # Get the number of truth boxes in each batch
        batch_size = preds.size(0)
        num_boxes = torch.tensor([t.size(0) for t in torch.split(target_bboxes, batch_size, dim=0)])

        # Calculate losses
        l_bbox = self.loss_bboxes(pred_bboxes, target_bboxes, num_boxes)
        l_giou = self.loss_giou(pred_bboxes, target_bboxes, num_boxes)
        l_class = self.loss_class(pred_probs, target_classes, num_boxes, pad_mask)

        loss = self.bbox_weight * l_bbox + self.giou_weight * l_giou + \
            self.class_weight * l_class
        loss /= batch_size
        return loss


#class DETRLoss(torch.nn.Module):
#    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
#        super().__init__()
#        self.num_classes = num_classes
#        self.matcher = matcher
#        self.weight_dict = weight_dict
#        self.eos_coef = eos_coef
#        self.losses = losses
#
#    @torch.no_grad()
#    def forward(self, outputs, targets):
#        # Placeholder for loss computation logic
#        batch_size, num_queries = outputs['pred_logits'].shape[:2]
#
#        cost_bbox = self.loss_bboxes
#
#    def loss_bboxes(self, outputs, targets, num_boxes):
#        # Placeholder for bounding box loss computation
#        giou = self.calc_giou(outputs['pred_boxes'], targets['boxes'])
#        l1 = torch.nn.functional.l1_loss(outputs['pred_boxes'], targets['boxes'][:, None, :, :], reduction='none')
#        giou_loss = 1 - giou
#        return l1.sum() / num_boxes, giou_loss.sum() / num_boxes
#
#    def calc_iou(self, boxes1, boxes2):
#        # Placeholder for IoU calculation logic
#        xcenter1, ycenter1, width1, height1 = boxes1[..., 0], boxes1[..., 1], boxes1[..., 2], boxes1[..., 3]
#        xcenter2, ycenter2, width2, height2 = boxes2[..., 0], boxes2[..., 1], boxes2[..., 2], boxes2[..., 3]
#
#        xmin1 = xcenter1 - 0.5 * width1
#        ymin1 = ycenter1 - 0.5 * height1
#        xmax1 = xcenter1 + 0.5 * width1
#        ymax1 = ycenter1 + 0.5 * height1
#
#        xmin2 = (xcenter2 - 0.5 * width2)[:, None, :, :]
#        ymin2 = (ycenter2 - 0.5 * height2)[:, None, :, :]
#        xmax2 = (xcenter2 + 0.5 * width2)[:, None, :, :]
#        ymax2 = (ycenter2 + 0.5 * height2)[:, None, :, :]
#
#        inter_xmin = torch.max(xmin1, xmin2)
#        inter_ymin = torch.max(ymin1, ymin2)
#        inter_xmax = torch.min(xmax1, xmax2)
#        inter_ymax = torch.min(ymax1, ymax2)
#
#        inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)
#
#        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
#        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
#
#        union_area = area1 + area2 - inter_area
#        iou = (inter_area / (union_area + 1e-6)).clamp(min=0., max=1.)
#        return iou, inter_area, union_area
#
#    def calc_giou(self, boxes1, boxes2):
#        # Placeholder for GIoU calculation logic
#        iou, inter_area, union_area = self.calc_iou(boxes1, boxes2)
#
#        xcenter1, ycenter1, width1, height1 = boxes1[..., 0], boxes1[..., 1], boxes1[..., 2], boxes1[..., 3]
#        xcenter2, ycenter2, width2, height2 = boxes2[..., 0], boxes2[..., 1], boxes2[..., 2], boxes2[..., 3]
#
#        xmin1 = xcenter1 - 0.5 * width1
#        ymin1 = ycenter1 - 0.5 * height1
#        xmax1 = xcenter1 + 0.5 * width1
#        ymax1 = ycenter1 + 0.5 * height1    
#
#        xmin2 = (xcenter2 - 0.5 * width2)[:, None, :, :]
#        ymin2 = (ycenter2 - 0.5 * height2)[:, None, :, :]
#        xmax2 = (xcenter2 + 0.5 * width2)[:, None, :, :]
#        ymax2 = (ycenter2 + 0.5 * height2)[:, None, :, :]
#
#        enclose_xmin = torch.min(xmin1, xmin2)
#        enclose_ymin = torch.min(ymin1, ymin2)
#        enclose_xmax = torch.max(xmax1, xmax2)
#        enclose_ymax = torch.max(ymax1, ymax2)
#
#        enclose_area = torch.clamp(enclose_xmax - enclose_xmin, min=0) * torch.clamp(enclose_ymax - enclose_ymin, min=0)
#
#        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
#        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
#
#        giou = iou - (enclose_area - (area1 + area2 - inter_area)) / (enclose_area + 1e-6)
#
#        return giou
#
#    def calc_class_loss(self, outputs, targets, indices, num_boxes):
#        # Placeholder for classification loss computation
#        src_logits = outputs['pred_logits']
#        idx = self._get_src_permutation_idx(indices)
#        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                    dtype=torch.int64, device=src_logits.device)
#        target_classes[idx] = target_classes_o
#
#        loss_ce = torch.nn.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, reduction='none')
#        return loss_ce.sum() / num_boxes
#
#
#class HungarianMatcher:
#    def __init__(self, criterion, cost_class=1., cost_bbox=1., cost_giou=1.):
#        self.criterion = criterion
#        self.cost_class = cost_class
#        self.cost_bbox = cost_bbox
#        self.cost_giou = cost_giou
#
#    def __call__(self, outputs, targets):
#        # Placeholder for matching logic
#        giou_cost, bbox_cost, class_cost = self.criterion.loss_bboxes(outputs, targets, None, 1)
#        total_cost = self.cost_bbox * bbox_cost + self.cost_giou * giou_cost + self.cost_class * class_cost
#
#        # Implement Hungarian algorithm here to find optimal assignment
#        row_minimum = total_cost.min(dim=2)
#        total_cost = total_cost - row_minimum.values.unsqueeze(2)
#        col_minimum = total_cost.min(dim=3)
#        total_cost = total_cost - col_minimum.values.unsqueeze(3)