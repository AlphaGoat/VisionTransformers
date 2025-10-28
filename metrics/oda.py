"""
Module for calculating ODA (Object Detection Analysis) metrics.

Author: Peter Thomas
Date: 24 October 2025
"""
import numpy as np
from typing import List, Dict


def perform_oda_evaluation(epoch_targets, epoch_outputs) -> List[Dict[str, float]]:
    """
    Formats ODA metrics into a readable string.

    Args:
        oda_metrics: A dictionary containing ODA metrics.

    Returns:
        A formatted string of ODA metrics.
    """
    image_dicts = []
    for i, batch_out in enumerate(epoch_outputs):
        for j, out in enumerate(batch_out):
            image_dict ={
                'image_id': i * len(batch_out) + j,
                'detections': out['detections'],
                'ground_truths': epoch_targets[i][j]['ground_truths']
            }
            image_dicts.append(image_dict)

    oda_metric_calculator = ODAMetric(image_dicts)
    oda_metrics = oda_metric_calculator.compute()

    formatted_metrics = "ODA Metrics:\n"
    for key, value in oda_metrics.items():
        formatted_metrics += f"{key}: {value:.4f}\n"
    return formatted_metrics


class ODAMetric:
    def __init__(self, image_dicts, confidence_thresholds=None, iou_thresholds=None, num_classes=80):
        self.reset()
        self.num_classes = num_classes
        if confidence_thresholds is None:
            confidence_thresholds = [0.1 * i for i in range(1, 10)]

        if iou_thresholds is None:
            iou_thresholds = [0.1 * i for i in range(1, 10)]

        analysis_dicts = []
        for i, img_dict in enumerate(image_dicts):
            confusion_matrix, tp, fp, fn = self.analyze(img_dict)
            analysis_dicts.append({
                'confusion_matrix': confusion_matrix,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            })

    def analyze(self, image_dict):
        """
        Per image analysis.
        """
        detections = image_dict['detections']
        ground_truths = image_dict['ground_truths']

        confusion_matrix = np.zeros((self.num_classes, self.num_classes, len(self.iou_thresholds), len(self.confidence_thresholds)), dtype=int)

        # Placeholder for actual analysis logic
        true_positives = []
        false_positives = []
        false_negatives = []

        # Iterate over confidence thresholds 
        for conf_threshold in self.confidence_thresholds:
            # Filter out detections lower than confidence threshold
            filtered_detections = self._filter_by_confidence_threshold(self.detections, conf_threshold)
            for det in filtered_detections:
                # Iterate over iou thresholds
                for iou_threshold in self.iou_thresholds:
                    # Filter out ground truths lower than iou threshold
                    match_found = False
                    for gt in self.ground_truths:
                        iou = self._calculate_iou(det['bbox'], gt['bbox'])
                        unassigned_ground_truths = self.ground_truths.copy()
                        if iou >= iou_threshold:
                            if det['label'] == gt['label']: # Match found
                                true_positives.append({'det': det, 'gt': gt, 'iou_threshold': iou_threshold, 'conf_threshold': conf_threshold})
                            else:
                                false_positives.append({'det': det, 'gt': gt, 'iou_threshold': iou_threshold, 'conf_threshold': conf_threshold})
                            match_found = True
                            confusion_matrix[gt['label'], det['label'], self.iou_thresholds.index(iou_threshold), self.confidence_thresholds.index(conf_threshold)] += 1
                            unassigned_ground_truths.remove(gt)

                    if not match_found:
                        false_positives.append({'det': det, 'gt': None, 'conf_threshold': conf_threshold, 'iou_threshold': iou_threshold})

                    for gt in unassigned_ground_truths:
                        false_negatives.append({'det': None, 'gt': gt, 'conf_threshold': conf_threshold, 'iou_threshold': iou_threshold})
                            # Further analysis can be done here based on IoU and thresholds

        return confusion_matrix, true_positives, false_positives, false_negatives
    
    def _filter_by_confidence_threshold(self, detections, threshold):
        """
        Filter detections based on a confidence threshold.

        Args:
            detections (list of dict): List of detection results.
            threshold (float): Confidence threshold.

        Returns:
            list of dict: Filtered detections.
        """
        return [det for det in detections if det['score'] >= threshold]

    def compute(self):
        """
        Compute the ODA metrics based on accumulated detections and ground truths.

        Returns:
            dict: A dictionary containing ODA metrics.
        """
        # Placeholder for actual ODA computation logic
        # Here we just return dummy values for illustration
        oda_metrics = {
            'precision': np.random.rand(),
            'recall': np.random.rand(),
            'f1_score': np.random.rand()
        }
        return oda_metrics

    def _calculate_iou(self, boxA, boxB):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Args:
            boxA (list): Bounding box A in the format [x1, y1, x2, y2].
            boxB (list): Bounding box B in the format [x1, y1, x2, y2].

        Returns:
            float: IoU value.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
