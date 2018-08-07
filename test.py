import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.misc as misc
import numpy as np
import os
import cv2
import scipy.io as sio
import sfcn_model

epsilon = 1e-7

def non_max_suppression(img, overlap_thresh=0.3, max_boxes=1200, r=8, prob_thresh=0.85):   #net_4_w6_di2.pkl
                                                                                               #over=0.2, max=1200,r=7,prob=0.85 --> P:0.837 R:0.894 F:0.865
                                                                                               #over=0.3, max=1200,r=8,prob=0.85 --> P:0.824 R:0.920 F:0.869
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    probs = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] < prob_thresh:
                img[i,j] = 0
            else:
                x1 = max(j - r, 0)
                y1 = max(i - r, 0)
                x2 = min(j + r, img.shape[1] - 1)
                y2 = min(i + r, img.shape[0] - 1)
                x1s.append(x1)
                y1s.append(y1)
                x2s.append(x2)
                y2s.append(y2)
                probs.append(img[i,j])
    x1s = np.array(x1s)
    y1s = np.array(y1s)
    x2s = np.array(x2s)
    y2s = np.array(y2s)
    #print(x1s.shape)
    boxes = np.concatenate((x1s.reshape((x1s.shape[0],1)), y1s.reshape((y1s.shape[0],1)), x2s.reshape((x2s.shape[0],1)), y2s.reshape((y2s.shape[0],1))),axis=1)
    #print(boxes.shape)
    probs = np.array(probs)
    pick = []
    area = (x2s - x1s) * (y2s - y1s)
    indexes = np.argsort([i for i in probs])

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        xx1_int = np.maximum(x1s[i], x1s[indexes[:last]])
        yy1_int = np.maximum(y1s[i], y1s[indexes[:last]])
        xx2_int = np.minimum(x2s[i], x2s[indexes[:last]])
        yy2_int = np.minimum(y2s[i], y2s[indexes[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int
        # find the union
        area_union = area[i] + area[indexes[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break
            # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick]
    #print(boxes.shape)

    return boxes
'''
def get_metrics(gt, pred, r=6):
    # calculate precise, recall and f1 score
    gt = np.array(gt).astype('int')
    if pred == []:
        if gt.shape[0] == 0:
            return 1, 1, 1, 0
        else:
            return 0, 0, 0, 0


    pred = np.array(pred).astype('int')

    temp = np.concatenate([gt, pred])

    if temp.shape[0] != 0:
        x_max = np.max(temp[:, 0]) + 1
        y_max = np.max(temp[:, 1]) + 1

        gt_map = np.zeros((y_max, x_max), dtype='int')
        for i in range(gt.shape[0]):
            x = gt[i, 0]
            y = gt[i, 1]
            x1 = max(0, x-r)
            y1 = max(0, y-r)
            x2 = min(x_max, x+r)
            y2 = min(y_max, y+r)
            gt_map[y1:y2,x1:x2] = 1

        pred_map = np.zeros((y_max, x_max), dtype='int')
        for i in range(pred.shape[0]):
            x = pred[i, 0]
            y = pred[i, 1]
            pred_map[y, x] = 1

        result_map = gt_map * pred_map
        tp = result_map.sum()

        precision = tp / (pred.shape[0] + epsilon)
        recall = tp / (gt.shape[0] + epsilon)
        f1_score = 2 * (precision * recall / (precision + recall + epsilon))

        return precision, recall, f1_score, tp
'''

def get_metrics(gt, pred, r=6):
    # calculate precise, recall and f1 score
    gt = np.array(gt).astype('int')
    if pred == []:
        if gt.shape[0] == 0:
            return 1, 1, 1, 0
        else:
            return 0, 0, 0, 0


    pred = np.array(pred).astype('int')

    temp = np.concatenate([gt, pred])

    if temp.shape[0] != 0:
        x_max = np.max(temp[:, 0]) + 1
        y_max = np.max(temp[:, 1]) + 1

        #gt_map = np.zeros((y_max, x_max), dtype='int')
        gt_map = np.zeros((500,500)).astype(np.int)
        for i in range(gt.shape[0]):
            x = gt[i, 0]
            y = gt[i, 1]
            x1 = max(0, x-r)
            y1 = max(0, y-r)
            x2 = min(x_max, x+r)
            y2 = min(y_max, y+r)
            #gt_map[y1:y2,x1:x2] = 1
            cv2.circle(gt_map, (x, y), r, 1, -1)
        #plt.imshow(gt_map)
        #plt.show()

        #pred_map = np.zeros((y_max, x_max), dtype='int')
        pred_map = np.zeros((500,500),dtype='int')
        for i in range(pred.shape[0]):
            x = pred[i, 0]
            y = pred[i, 1]
            pred_map[y, x] = 1

        result_map = gt_map * pred_map
        tp = result_map.sum()

        precision = min(tp / (pred.shape[0] + epsilon),1)
        recall = min(tp / (gt.shape[0] + epsilon),1)
        f1_score = 2 * (precision * recall / (precision + recall + epsilon))

        return precision, recall, f1_score, tp

def test_det(model):
    path = './aug'
    # path = './data/test'

    tp_num = 0
    gt_num = 0
    pred_num = 0
    for i in range(81,101):
        filename = os.path.join(path,'img'+str(i)+'_1.bmp')
        if os.path.exists(filename):
            gtpath = 'F:/CRCHistoPhenotypes_2016_04_28/Detection'
            imgname = 'img' + str(i)
            print(imgname)

            # === load model =========================
            #model = det_model.Net_2()
            #model = net4.Net_4_di()
            #model = net_di.Net_4_di()
            #model = sfcn_model.Attention_Net()
            #model.load_state_dict(torch.load('./ckpt/test_att.pkl'))
            model = model.cuda()
            #print(model)

            # === test image =========================
            img = misc.imread(filename)
            outputbase = img.copy()
            # === preprocess =========================
            # crop the image into small pathces to fit the model

            #img = img / 255.
            #img -= np.mean(img,keepdims=True)
            #img /= (np.std(img,keepdims=True) + 1e-7)
            img = misc.imresize(img,(256,256))

            img = img - 128.
            img = img / 128.
            img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
            img = np.transpose(img,(0,3,1,2))
            img = torch.Tensor(img).cuda()
            #patches = crop_to_pathces(img)
            #patches = np.transpose(patches,(0,3,1,2))
            #patches = torch.Tensor(patches).cuda()
            #print(patches.shape)

            # === show result =========================
            # concatenate the patches
            #det = model.get_layer(name='thresholded_re_lu_1').output
            #detmodel = Model(inputs=model.inputs,outputs=det)
            result = model(img)[0]
            #print(result)
            #print(result.shape)
            result = result.cpu().detach().numpy()
            #print(result.shape)
            result = np.transpose(result,(0,2,3,1))[0]
            #result = result[:,:,:,1]
            result = np.exp(result)
            for p in range(1,result.shape[-1]):
                result_cls = result[:,:,p]
                result_cls = misc.imresize(result_cls,(500,500))
                result_cls = result_cls / 255.
                #print(result)
                #print(result.shape)

                #plt.imshow(result_cls)
                #plt.colorbar()
                #plt.show()
                boxes = non_max_suppression(result_cls)
                num_of_nuclei = boxes.shape[0]
                print('detection:', num_of_nuclei)
                matname = imgname + '_detection.mat'
                #matname = imgname + '_detection.mat'
                matpath = os.path.join(gtpath, imgname, matname)
                gt = sio.loadmat(matpath)['detection']
                print('gt:',gt.shape[0])
                for i in range(gt.shape[0]):
                    cv2.circle(outputbase, (int(gt[i, 0]), int(gt[i, 1])), 6 , (255,255,0), 1)
                pred = []
                for j in range(boxes.shape[0]):
                    x1 = boxes[j, 0]
                    y1 = boxes[j, 1]
                    x2 = boxes[j, 2]
                    y2 = boxes[j, 3]
                    cx = int(x1 + (x2 - x1) / 2)
                    cy = int(y1 + (y2 - y1) / 2)
                    #cv2.rectangle(outputbase,(x1, y1), (x2, y2),(255,0,0), 1)
                    cv2.circle(outputbase, (cx, cy), 2, (255,255,0), -1)
                    pred.append([cx, cy])
                p, r, f1, tp = get_metrics(gt, pred)

                print(p, r, f1)
                tp_num += tp
                gt_num += gt.shape[0]
                pred_num += np.array(pred).shape[0]

                #plt.imshow(outputbase)
                #plt.colorbar()
                #plt.show()

    precision = tp_num / (pred_num + epsilon)
    recall = tp_num / (gt_num + epsilon)
    f1_score = 2 * (precision * recall / (precision + recall + epsilon))

    return tp_num, pred_num, gt_num, precision, recall, f1_score

def test_cls(model):
    path = './aug'
    # path = './data/test'

    tp_num = 0
    gt_num = 0
    pred_num = 0
    type_group = ['epithelial', 'fibroblast', 'inflammatory', 'others']
    color_group = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    model = model.cuda()
    for i in range(81,101):
        filename = os.path.join(path,'img'+str(i)+'_1.bmp')
        if os.path.exists(filename):
            gtpath = 'F:/CRCHistoPhenotypes_2016_04_28/Classification'
            imgname = 'img' + str(i)
            print(imgname)

            img = misc.imread(filename)
            img1 = img.copy()
            img2 = img.copy()
            img = misc.imresize(img,(256,256))
            img = img - 128.
            img = img / 128.
            img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
            img = np.transpose(img,(0,3,1,2))
            img = torch.Tensor(img).cuda()

            # === show result =========================
            result = model(img)[1]
            result = result.cpu().detach().numpy()
            result = np.transpose(result, (0, 2, 3, 1))[0]
            result = np.exp(result)
            arg_map = np.argmax(result, axis=-1)
            r_map = np.zeros((result.shape[0], result.shape[1], 5))
            for a in range(arg_map.shape[0]):
                for b in range(arg_map.shape[1]):
                    type_index = arg_map[a, b]
                    r_map[a, b, type_index] = 1
            result = result * r_map
            for p in range(1,result.shape[-1]):
                outputbase = img1.copy()
                result_cls = result[:,:,p]
                result_cls = misc.imresize(result_cls, (500,500), interp='nearest')
                result_cls = result_cls / 255.
                #print(result)
                #print(result.shape)

                #plt.imshow(result_cls)
                #plt.colorbar()
                #plt.show()
                boxes = non_max_suppression(result_cls, prob_thresh=0.4, overlap_thresh=0.1)
                num_of_nuclei = boxes.shape[0]
                print('detection:', num_of_nuclei)
                matname = imgname + '_' + type_group[p-1] + '.mat'
                matpath = os.path.join(gtpath, imgname, matname)
                gt = sio.loadmat(matpath)['detection']
                print('gt:',gt.shape[0])
                #for i in range(gt.shape[0]):
                    #cv2.circle(outputbase, (int(gt[i, 0]), int(gt[i, 1])), 6 , (255,255,0), 1)
                pred = []
                for j in range(boxes.shape[0]):
                    x1 = boxes[j, 0]
                    y1 = boxes[j, 1]
                    x2 = boxes[j, 2]
                    y2 = boxes[j, 3]
                    cx = int(x1 + (x2 - x1) / 2)
                    cy = int(y1 + (y2 - y1) / 2)
                    #cv2.rectangle(outputbase,(x1, y1), (x2, y2),(255,0,0), 1)
                    cv2.circle(img2, (cx, cy), 3, color_group[p-1], -1)
                    pred.append([cx, cy])
                p, r, f1, tp = get_metrics(gt, pred)

                print(p, r, f1)
                tp_num += tp
                gt_num += gt.shape[0]
                pred_num += np.array(pred).shape[0]

            #plt.imshow(img2)
            #plt.colorbar()
            #plt.show()

    precision = tp_num / (pred_num + epsilon)
    recall = tp_num / (gt_num + epsilon)
    f1_score = 2 * (precision * recall / (precision + recall + epsilon))

    return tp_num, pred_num, gt_num, precision, recall, f1_score

def show_results(model):
    path = './aug'
    # path = './data/test'

    tp_num = 0
    gt_num = 0
    pred_num = 0
    type_group = ['epithelial', 'fibroblast', 'inflammatory', 'others']
    model = model.cuda()
    for i in range(81,101):
        filename = os.path.join(path,'img'+str(i)+'_1.bmp')
        if os.path.exists(filename):
            gtpath = 'F:/CRCHistoPhenotypes_2016_04_28/Classification'
            imgname = 'img' + str(i)
            print(imgname)

            img = misc.imread(filename)
            img1 = img.copy()
            img = misc.imresize(img,(256,256))
            img = img - 128.
            img = img / 128.
            img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
            img = np.transpose(img,(0,3,1,2))
            img = torch.Tensor(img).cuda()

            # === show result =========================
            '''
            result = model(img)[1]
            result = result.cpu().detach().numpy()
            result = np.transpose(result,(0,2,3,1))[0]
            result = np.exp(result)
            result = np.argmax(result,axis=-1)
            plt.imshow(result)
            plt.show()
            '''
            result = model(img)[2]
            result = result.cpu().detach().numpy()
            result = np.transpose(result, (0, 2, 3, 1))[0]
            result = np.reshape(result,(result.shape[0],result.shape[1]))
            print(result.shape)
            plt.imshow(result)
            plt.colorbar()
            plt.show()

model = sfcn_model.Attention_Net()
model.load_state_dict(torch.load('./ckpt/test_att32.pkl'))

show_results(model)

#tp_num, pred_num, gt_num, precision, recall, f1_score = test_cls(model)
'''
print(tp_num)
print(pred_num)
print(gt_num)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1_score)
'''

