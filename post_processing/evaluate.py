import cv2
import numpy as np
epsilon = 1e-7

def get_metrics(gt, pred, r=6):
    # calculate precise, recall and f1 score

    gt = np.array(gt).astype('int')
    pred = np.array(pred).astype('int')

    gt_map = np.zeros((500,500)).astype(np.int)
    pred_map = np.zeros((500,500)).astype(np.int)

    for i in range(gt.shape[0]):
        x = gt[i, 0]
        y = gt[i, 1]
        cv2.circle(gt_map, (x, y), r , 1, -1)

    for i in range(pred.shape[0]):
        x = pred[i, 0]
        y = pred[i, 1]
        pred_map[y, x] = 1

    result_map = gt_map * pred_map
    tp = result_map.sum()

    precision = min(tp / (pred.shape[0] + epsilon), 1)
    recall = min(tp / (gt.shape[0] + epsilon), 1)
    f1_score = 2 * (precision * recall / (precision + recall + epsilon))

    return precision, recall, f1_score, tp