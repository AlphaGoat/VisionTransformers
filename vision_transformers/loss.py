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
    def __init__(self, batch_size, num_classes, class_weight=1.0, giou_weight=1.0, bbox_weight=1.0):
        self.batch_size = batch_size
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
        pred_bboxes = pred_bboxes #.reshape(-1, 4) # [batch_size * num_queries, 4]
        pred_probs = pred_probs.reshape(-1, self.num_classes) # [batch_size * num_queries, num_classes]

        # Prepare targets
        targets = self._pad_targets(targets, num_queries)
        target_bboxes = targets["boxes"] #.reshape(-1, 4)  # [total_target_boxes, 4]
        target_classes = targets["labels"].reshape(-1)  # [total_target_boxes]

        # Class cost is negative of pred probability given for target class
        C_classes = -1. * torch.gather(pred_probs, 1, target_classes[:, None]).squeeze(1)
        C_classes = C_classes.reshape(batch_size, -1, 1)

        # L1 norm for bounding box cost
        C_boxes = torch.cdist(pred_bboxes, target_bboxes, p=1)

#        C_giou = -ops.generalized_box_iou(
#            ops.box_convert(pred_bboxes, in_fmt='cxcywh', out_fmt='xyxy'),
#            ops.box_convert(target_bboxes, in_fmt='cxcywh', out_fmt='xyxy'),
#        )
        C_giou = -self._generalized_box_iou(
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

        return [(torch.from_numpy(i), torch.from_numpy(j)) for i, j in indices]

    def loss_bboxes(self, p_bboxes, t_bboxes, n_boxes):
        # Bbox loss is just L1 loss
        loss_bbox = F.l1_loss(p_bboxes, t_bboxes, reduction="none")

        # Normalize loss by number of boxes in each batch
        import pdb; pdb.set_trace()
        loss_bbox = loss_bbox.sum(dim=2)
        loss_bbox /= n_boxes
        return loss_bbox

    def loss_giou(self, p_bboxes, t_bboxes, n_boxes):
        p_bboxes = p_bboxes.reshape(-1, 4)
        t_bboxes = t_bboxes.reshape(-1, 4)
        loss_giou = 1. - torch.diag(
            ops.generalized_box_iou(
                ops.box_convert(p_bboxes, in_fmt='cxcywh', out_fmt='xyxy'),
                ops.box_convert(t_bboxes, in_fmt='cxcywh', out_fmt='xyxy'),
            )
        )
        loss_giou = loss_giou.reshape(self.batch_size, -1)
        loss_giou = loss_giou.sum(dim=1)
        loss_giou /= n_boxes
        return loss_giou

    def loss_class(self, p_probs, t_classes, n_boxes):
        loss_class = torch.nn.CrossEntropyLoss()(p_probs, t_classes, reduction="none")
        import pdb; pdb.set_trace()
        loss_class = loss_class.reshape(self.batch_size, -1)
        loss_class /= n_boxes
        return loss_class

    def __call__(self, preds, targets):
        batch_size = len(targets)
        num_queries = preds["pred_boxes"].size(1)

        # Hungarian matching to find optimum pairs
        indices = self.hungarian_matcher(preds, targets)
        pred_idxs = torch.cat([i for i, j in indices]).reshape(batch_size, -1)
        target_idxs = torch.cat([j for i, j in indices]).reshape(batch_size, -1)

        # Prepare targets by padding
        targets = self._pad_targets(targets, preds["pred_boxes"].size(1))

        # Order target indices (makes tracking a little easier...)
        pred_idxs = torch.gather(pred_idxs, 1, target_idxs.argsort(dim=1))

        # Unpack predictions and targets
        pred_bboxes = preds["pred_boxes"]
        target_bboxes = targets["boxes"]

        pred_probs = preds["pred_logits"]
        target_classes = targets["labels"]

        # Get the number of truth boxes in each batch
        num_boxes = num_queries

        # Calculate losses
        l_bbox = self.loss_bboxes(pred_bboxes, target_bboxes, num_boxes)
        l_giou = self.loss_giou(pred_bboxes, target_bboxes, num_boxes)
        l_class = self.loss_class(pred_probs, target_classes, num_boxes)

        loss = self.bbox_weight * l_bbox + self.giou_weight * l_giou + \
            self.class_weight * l_class
        loss /= batch_size
        return loss

    def _generalized_box_iou(self, boxes1, boxes2):
        """ Compute generalized IoU between two sets of boxes. 
        Args:
            boxes1 (torch.Tensor): Tensor of shape (N, M, 4) in (x0, y0, x1, y1) format.
            boxes2 (torch.Tensor): Tensor of shape (N, K, 4) in (x0, y0, x1, y1) format.
        """

        # Calculate the intersection between boxes
        gious = []
        for b in range(self.batch_size):
            b1 = boxes1[b]
            b2 = boxes2[b]
            x_min = torch.max(b1[:, None, 0], b2[:, 0])
            y_min = torch.max(b1[:, None, 1], b2[:, 1])
            x_max = torch.min(b1[:, None, 2], b2[:, 2])
            y_max = torch.min(b1[:, None, 3], b2[:, 3])

            inter_area = torch.clamp(x_max - x_min, min=0) * torch.clamp(y_max - y_min, min=0)

            # Calculate the area of each box
            area1 = (b1[:, None, 2] - b1[:, 0]) * (b1[:, None, 3] - b1[..., 1])
            area2 = (b2[:, None, 2] - b2[:, 0]) * (b2[:, None, 3] - b2[..., 1])

            union_area = area1 + area2 - inter_area
            iou = (inter_area / (union_area + 1e-6)).clamp(min=0., max=1.)

            # Calculate the area of the smallest enclosing box
            enclose_x_min = torch.min(b1[:, None, 0], b2[:, 0])
            enclose_y_min = torch.min(b1[:, None, 1], b2[:, 1])
            enclose_x_max = torch.max(b1[:, None, 2], b2[:, 2])
            enclose_y_max = torch.max(b1[:, None, 3], b2[:, 3])

            enclose_area = torch.clamp(enclose_x_max - enclose_x_min, min=0) * torch.clamp(enclose_y_max - enclose_y_min, min=0)

            # Calculate generalized IoU
            gious.append(iou - (enclose_area - union_area) / (enclose_area + 1e-6))

        gious = torch.stack(gious, dim=0) 
        return gious

    @staticmethod
    def _pad_targets(targets, num_queries):
        """ Pad target boxes and classes to match number of queries. """
        target_bboxes = [t["boxes"] for t in targets]
        target_classes = [t["labels"] for t in targets]

        padded_boxes = torch.stack([F.pad(bboxes, (0, 0, 0, num_queries - bboxes.size(0)), value=0) for bboxes in target_bboxes])
        padded_labels = torch.stack([F.pad(labels, (0, num_queries - labels.size(0)), value=0) for labels in target_classes])

        return {"boxes": padded_boxes, "labels": padded_labels}

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