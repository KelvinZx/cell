from cell_detection_model import CellDetection
from tensorboardX import SummaryWriter
from post_processing.nms import non_max_suppression
from post_processing.evaluate import get_metrics
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import dataset
from config import Config
import visdom
import os
import scipy.misc as misc
import scipy.io as sio
import cv2
import math
from dataset import load_data
from util import DATA_DIR

BATCH_SIZE = Config.image_per_gpu * Config.gpu_count
epsilon = 1e-7
TARGET_SIZE = 224


def MyMetrics(model,target_size=256):
    path = './aug'
    tp_num = [0, 0]
    gt_num = [0, 0]
    pred_num = [0, 0]
    precision = [0, 0]
    recall = [0, 0]
    f1_score = [0, 0]
    cell_type_group = ['epithelial', 'fibroblast', 'inflammatory', 'others']
    gt_type = ['Detection', 'Classification']

    for i in range(81, 101):
        filename = os.path.join(path, 'img' + str(i) + '_1.bmp')
        imgname = 'img' + str(i)
        if os.path.exists(filename):
            img = misc.imread(filename)
            img = misc.imresize(img, (target_size, target_size))
            img = img - 128.
            img = img / 128.
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = np.transpose(img, (0, 3, 1, 2))
            img = torch.Tensor(img).cuda()
            img_result = model(img)

            for index, ground_truth in enumerate(gt_type, 0):
                gtpath = os.path.join('F:/CRCHistoPhenotypes_2016_04_28',ground_truth)
                result = img_result[index]
                result = result.cpu().detach().numpy()
                result = np.transpose(result, (0, 2, 3, 1))[0]
                result = np.exp(result)
                for j in range(1, result.shape[-1]):
                    result_type = result[:, :, j]
                    result_type = misc.imresize(result_type, (500, 500))
                    result_type = result_type / 255.
                    boxes = non_max_suppression(result_type)
                    if ground_truth == 'Detection':
                        matname = imgname + '_detection.mat'
                    else:
                        matname = imgname + '_' + cell_type_group[j-1] + '.mat'
                    matpath = os.path.join(gtpath, imgname, matname)
                    gt = sio.loadmat(matpath)['detection']
                    pred = []
                    for k in range(boxes.shape[0]):
                        x1 = boxes[k, 0]
                        y1 = boxes[k, 1]
                        x2 = boxes[k, 2]
                        y2 = boxes[k, 3]
                        cx = int(x1 + (x2 - x1) / 2)
                        cy = int(y1 + (y2 - y1) / 2)
                        pred.append([cx, cy])
                    p, r, f1, tp = get_metrics(gt, pred)
                    tp_num[index] += tp
                    gt_num[index] += gt.shape[0]
                    pred_num[index] += np.array(pred).shape[0]

    for index, ground_truth in enumerate(gt_type, 0):
        precision[index] = tp_num[index] / (pred_num[index] + epsilon)
        recall[index] = tp_num[index] / (gt_num[index] + epsilon)
        f1_score[index] = 2 * (precision[index] * recall[index] / (precision[index] + recall[index] + epsilon))

    return precision, recall, f1_score


def train(model, weight_det=None, weight_cls=None,data_dir='',
          preprocess=True, gpu=True, num_epochs=Config.epoch, target_size=256):
    if weight_det == None:
        weight_det = torch.Tensor([1, 1])
    else:
        weight_det = torch.Tensor(weight_det)

    if weight_cls == None:
        weight_cls = torch.Tensor([1, 1, 1, 1, 1])
    else:
        weight_cls = torch.Tensor(weight_cls)

    data = dataset.CRC_joint(data_dir, target_size=target_size)

    x_train, y_train_det, y_train_cls = load_data(DATA_DIR, 'train', reshape_size=(256,256))
    #x_train, y_train_det, y_train_cls = data.load_train(preprocess=preprocess)
    train_count = len(x_train)
    train_steps = math.ceil(train_count / BATCH_SIZE)
    x_val, y_val_det, y_val_cls = load_data(DATA_DIR, 'validation', reshape_size=(256, 256))
    #x_val, y_val_det, y_val_cls = data.load_val(preprocess=preprocess)
    val_count = len(x_val)
    val_steps = int(val_count / BATCH_SIZE)
    print('training imgs:', train_count)
    print('val imgs:', val_count)

    trainset = np.concatenate([x_train, y_train_det, y_train_cls], axis=1)
    trainset = torch.Tensor(trainset)

    valset = np.concatenate([x_val, y_val_det, y_val_cls], axis=1)
    valset = torch.Tensor(valset)

    if gpu:
        model = model.cuda()
        trainset = trainset.cuda()
        valset = valset.cuda()
        weight_det = weight_det.cuda()
        weight_cls = weight_cls.cuda()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
    NLLLoss_det = nn.NLLLoss(weight=weight_det)
    NLLLoss_cls = nn.NLLLoss(weight=weight_cls)
    best_loss = 99999.0
    train_loss_list = []
    train_step_list = []
    val_loss_list = []
    val_det_p = []
    val_det_r = []
    val_det_f = []
    val_cls_p = []
    val_cls_r = []
    val_cls_f = []
    epoch_list = []
    vis = visdom.Visdom(env=u'det')
    vis_joint = visdom.Visdom(env=u'joint')

    for epoch in range(num_epochs):

        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        for i, datapack in enumerate(train_loader, 0):
            train_imgs = datapack[:, 0:3]
            train_det_masks = datapack[:, 3:4]
            train_cls_masks = datapack[:, 4:]

            train_det_masks = train_det_masks.long()

            train_det_masks = train_det_masks.view(
                train_det_masks.size()[0],
                train_det_masks.size()[2],
                train_det_masks.size()[3]
            )


            optimizer.zero_grad()
            train_det_out= model(train_imgs)
            t_det_loss = NLLLoss_det(train_det_out, train_det_masks)
            #t_cls_loss = NLLLoss_cls(train_cls_out, train_cls_masks)
            t_loss = t_det_loss #+ t_cls_loss
            t_loss.backward()
            optimizer.step()
            train_loss += t_loss.item()

            if i % 10 == 9:
                print('det epoch: %3d, step: %3d loss: %.5f' % (epoch + 1, i + 1, train_loss / 10))
                train_loss_list.append(train_loss/10)
                train_step_list.append((train_steps * epoch + i + 1))

                trace_p = dict(x=train_step_list, y=train_loss_list, mode="lines", type='custom', name='train_loss')
                layout = dict(title="train loss", xaxis={'title': 'step'}, yaxis={'title': 'loss'})

                vis._send({'data': [trace_p], 'layout': layout, 'win': 'trainloss'})
                train_loss = 0.0

        for i, datapack in enumerate(val_loader, 0):
            val_imgs = datapack[:, 0:3]
            val_det_masks = datapack[:, 3:4]
            val_cls_masks = datapack[:, 4:]

            val_det_masks = val_det_masks.long()
            val_det_masks = val_det_masks.view(
                val_det_masks.size()[0],
                val_det_masks.size()[2],
                val_det_masks.size()[3]
            )

            # optimizer.zero_grad()
            val_det_out = model(val_imgs)
            v_det_loss = NLLLoss_det(val_det_out, val_det_masks)
            v_loss = v_det_loss
            val_loss += v_loss.item()

            if i % val_steps == val_steps - 1:
                val_loss = val_loss / val_steps
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), './ckpt/test_att_global.pkl')
                end = time.time()
                time_spent = end - start
                print('det epoch: %3d, time: %.5f val_loss: %.5f' % (epoch + 1, time_spent, val_loss))
                val_loss_list.append(val_loss)
                epoch_list.append(epoch + 1)

                trace_p = dict(x=epoch_list, y=val_loss_list, mode='lines', type='custom', name='val_loss')
                layout = dict(title='val loss', xaxis={'title': 'epoch'}, yaxis={'title': 'loss'})
                vis._send({'data': [trace_p], 'layout': layout, 'win': 'valloss'})

                val_loss = 0.0
                p, r, f = MyMetrics(model, target_size=target_size)
                print('p:', p)
                print('r:', r)
                print('f:', f)
                val_det_p.append(p[0])
                val_det_r.append(r[0])
                val_det_f.append(f[0])
                val_cls_p.append(p[1])
                val_cls_r.append(r[1])
                val_cls_f.append(f[1])
                trace_det_f = dict(x=epoch_list, y=val_det_f, mode='lines', type='custom', name='val_det_f')
                layout = dict(title='val det f', xaxis={'title': 'epoch'}, yaxis={'title': 'F'})
                vis._send({'data': [trace_det_f], 'layout': layout, 'win': 'valdetf'})
                trace_det_p = dict(x=epoch_list, y=val_det_p, mode='lines', type='custom', name='val_det_p')
                trace_det_r = dict(x=epoch_list, y=val_det_r, mode='lines', type='custom', name='val_det_r')
                layout = dict(title='val det pr', xaxis={'title': 'epoch'}, yaxis={'title': 'PR'})
                vis._send({'data': [trace_det_p,trace_det_r], 'layout': layout, 'win': 'valdetpr'})
                trace_p = dict(x=epoch_list, y=val_cls_f, mode='lines', type='custom', name='val_cls_f')
                layout = dict(title='val cls f', xaxis={'title': 'epoch'}, yaxis={'title': 'F'})
                vis._send({'data': [trace_p], 'layout': layout, 'win': 'valclsf'})
                trace_cls_p = dict(x=epoch_list, y=val_cls_p, mode='lines', type='custom', name='val_cls_p')
                trace_cls_r = dict(x=epoch_list, y=val_cls_r, mode='lines', type='custom', name='val_cls_r')
                layout = dict(title='val cls pr', xaxis={'title': 'epoch'}, yaxis={'title': 'PR'})
                vis._send({'data': [trace_cls_p, trace_cls_r], 'layout': layout, 'win': 'valclspr'})
                print('******************************************************************************')

    for epoch in range(num_epochs):

        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        for i, datapack in enumerate(train_loader, 0):
            train_imgs = datapack[:, 0:3]
            train_det_masks = datapack[:, 3:4]
            train_cls_masks = datapack[:, 4:]

            train_det_masks = train_det_masks.long()
            train_cls_masks = train_cls_masks.long()

            train_det_masks = train_det_masks.view(
                train_det_masks.size()[0],
                train_det_masks.size()[2],
                train_det_masks.size()[3]
            )

            train_cls_masks = train_cls_masks.view(
                train_cls_masks.size()[0],
                train_cls_masks.size()[2],
                train_cls_masks.size()[3]
            )

            optimizer.zero_grad()
            train_det_out, train_cls_out = model(train_imgs)
            t_det_loss = NLLLoss_det(train_det_out, train_det_masks)
            t_cls_loss = NLLLoss_cls(train_cls_out, train_cls_masks)
            t_loss = 0.3 * t_det_loss + 0.7 * t_cls_loss
            t_loss.backward()
            optimizer.step()
            train_loss += t_loss.item()

            if i % 10 == 9:
                print('joint epoch: %3d, step: %3d loss: %.5f' % (epoch + 1, i + 1, train_loss / 10))
                train_loss_list.append(train_loss/10)
                train_step_list.append((train_steps * epoch + i + 1))

                trace_p = dict(x=train_step_list, y=train_loss_list, mode="lines", type='custom', name='train_loss')
                layout = dict(title="train loss", xaxis={'title': 'step'}, yaxis={'title': 'loss'})

                vis_joint._send({'data': [trace_p], 'layout': layout, 'win': 'trainloss'})
                train_loss = 0.0

        for i, datapack in enumerate(val_loader, 0):
            val_imgs = datapack[:, 0:3]
            val_det_masks = datapack[:, 3:4]
            val_cls_masks = datapack[:, 4:]

            val_det_masks = val_det_masks.long()
            val_det_masks = val_det_masks.view(
                val_det_masks.size()[0],
                val_det_masks.size()[2],
                val_det_masks.size()[3]
            )

            val_cls_masks = val_cls_masks.long()
            val_cls_masks = val_cls_masks.view(
                val_cls_masks.size()[0],
                val_cls_masks.size()[2],
                val_cls_masks.size()[3]
            )

            # optimizer.zero_grad()
            val_det_out, val_cls_out = model(val_imgs)
            v_det_loss = NLLLoss_det(val_det_out, val_det_masks)
            v_cls_loss = NLLLoss_cls(val_cls_out, val_cls_masks)
            v_loss = v_det_loss + v_cls_loss
            val_loss += v_loss.item()

            if i % val_steps == val_steps - 1:
                val_loss = val_loss / val_steps
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), './ckpt/test_att_global.pkl')
                end = time.time()
                time_spent = end - start
                print('joint epoch: %3d, time: %.5f val_loss: %.5f' % (epoch + 1, time_spent, val_loss))
                val_loss_list.append(val_loss)
                epoch_list.append(epoch + 1)

                trace_p = dict(x=epoch_list, y=val_loss_list, mode='lines', type='custom', name='val_loss')
                layout = dict(title='val loss', xaxis={'title': 'epoch'}, yaxis={'title': 'loss'})
                vis_joint._send({'data': [trace_p], 'layout': layout, 'win': 'valloss'})

                val_loss = 0.0
                p, r, f = MyMetrics(model, target_size=target_size)
                print('p:', p)
                print('r:', r)
                print('f:', f)
                val_det_p.append(p[0])
                val_det_r.append(r[0])
                val_det_f.append(f[0])
                val_cls_p.append(p[1])
                val_cls_r.append(r[1])
                val_cls_f.append(f[1])
                trace_det_f = dict(x=epoch_list, y=val_det_f, mode='lines', type='custom', name='val_det_f')
                layout = dict(title='val det f', xaxis={'title': 'epoch'}, yaxis={'title': 'F'})
                vis_joint._send({'data': [trace_det_f], 'layout': layout, 'win': 'valdetf'})
                trace_det_p = dict(x=epoch_list, y=val_det_p, mode='lines', type='custom', name='val_det_p')
                trace_det_r = dict(x=epoch_list, y=val_det_r, mode='lines', type='custom', name='val_det_r')
                layout = dict(title='val det pr', xaxis={'title': 'epoch'}, yaxis={'title': 'PR'})
                vis_joint._send({'data': [trace_det_p,trace_det_r], 'layout': layout, 'win': 'valdetpr'})
                trace_p = dict(x=epoch_list, y=val_cls_f, mode='lines', type='custom', name='val_cls_f')
                layout = dict(title='val cls f', xaxis={'title': 'epoch'}, yaxis={'title': 'F'})
                vis_joint._send({'data': [trace_p], 'layout': layout, 'win': 'valclsf'})
                trace_cls_p = dict(x=epoch_list, y=val_cls_p, mode='lines', type='custom', name='val_cls_p')
                trace_cls_r = dict(x=epoch_list, y=val_cls_r, mode='lines', type='custom', name='val_cls_r')
                layout = dict(title='val cls pr', xaxis={'title': 'epoch'}, yaxis={'title': 'PR'})
                vis_joint._send({'data': [trace_cls_p, trace_cls_r], 'layout': layout, 'win': 'valclspr'})
                print('******************************************************************************')

if __name__ == '__main__':
    net = CellDetection()
    train(net, weight_det=[0.1, 2], weight_cls=[0.1, 4, 3, 6, 10], data_dir='./aug', target_size=TARGET_SIZE, num_epochs=200)
