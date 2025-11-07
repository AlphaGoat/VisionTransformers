"""
Deformable-DETR loss function.

Author: Peter Thomas
Date: 07 October 2025
"""
import torch
from torchvision import ops
from scipy.optimize import linear_sum_assignment


class DETRLoss():
    def __init__(self, num_classes, weight_dict):
        self.weight_dict = weight_dict
        pass

    def hungarian_matcher(self, preds, targets):
        pred_bboxes = preds["pred_bboxes"]
        target_bboxes = targets["bboxes"]

        pred_classes = preds["class_logits"]
        target_classes = targets["labels"]

        p_probs = pred_classes.softmax(dim=-1)

        # Class cost is negative of pred probability given for target class
        C_classes = -1. * p_probs[..., target_classes]

        # L1 norm for bounding box cost
        C_boxes = torch.cdist(pred_bboxes, target_bboxes, p=1)

        C_giou = -ops.generalized_box_iou(
            ops.box_convert(pred_bboxes, in_fmt='cxcywh', out_fmt='xyxy'),
            ops.box_convert(target_bboxes, in_fmt='cxcywh', out_fmt='xyxy'),
        )

        C_total = self.weight_dict["cls_weight"] * C_classes + \
            self.weight_dict["box_weight"] * C_boxes + \
            self.weight_dict["giou_weight"] * C_giou

        C_total = C_total.cpu().detach().numpy()

        # Find the optimum pairs that produce the minimum summation
        pred_idxs, target_idxs = linear_sum_assignment(C_total)

        return C_total


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