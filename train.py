import numpy as np
import math
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import time
import cv2
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from utils import _potsdam_dsm_id
import Model.cfg as cfg
from torch.autograd import Variable
from IPython.display import clear_output
from MMNet import MMNet as MFNet
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

net = MFNet(num_classes=N_CLASSES).cuda()

def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('All Params:   ', params)

params1 = 0
params_lora = 0
params_adapter = 0
params_peft = 0
for name, param in net.image_encoder.named_parameters():
    if "peft_" in name:
        params_peft += param.nelement()
    elif "lora_" in name:
        params_lora += param.nelement()
    elif "Adapter" in name:
        params_adapter += param.nelement()
    else:
        params1 += param.nelement()
print('ImgEncoder:   ', params1)
print('PEFT: ', params_peft)
print('Lora: ', params_lora)
print('Adapter: ', params_adapter)
print('Others: ', params-params1-params_peft-params_lora-params_adapter)

# for name, parms in net.named_parameters():
#     print('%-50s' % name, '%-30s' % str(parms.shape), '%-10s' % str(parms.nelement()))

# params = 0
# for name, param in net.sam.prompt_encoder.named_parameters():
#     params += param.nelement()
# print('prompt_encoder: ', params)

# params = 0
# for name, param in net.sam.mask_decoder.named_parameters():
#     params += param.nelement()
# print('mask_decoder: ', params)

# print(net)

if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
    print("Using DataParallel with {} GPUs".format(torch.cuda.device_count()))

print("training : ", len(train_ids))
print("testing : ", len(test_ids))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 2e-4
weight_decay = 0.05
warmup_epochs = min(5, epochs)
min_lr = 1e-6

def build_param_groups(model, weight_decay):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

optimizer = optim.AdamW(
    build_param_groups(net, weight_decay),
    lr=base_lr,
    betas=(0.9, 0.999),
)

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_factor = min_lr / base_lr
    return min_factor + (1.0 - min_factor) * cosine

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    if DATASET == 'Potsdam':
        test_images = (normalize_rgb(1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32')) for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (normalize_rgb(1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32')) for id in test_ids)
    if DATASET == 'Potsdam':
        test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(_potsdam_dsm_id(id))), dtype='float32') for id in test_ids)
    else:
        test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    eval_labels = (convert_from_color(io.imread(LABEL_FOLDER.format(id))) for id in test_ids)

    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        for img, dsm, gt_e in tqdm(zip(test_images, test_dsms, eval_labels), total=len(test_ids), leave=False):
            if dsm.shape[:2] != img.shape[:2]:
                dsm = match_spatial_shape(dsm, img.shape)
            dsm_min = np.min(dsm)
            dsm_max = np.max(dsm)
            if dsm_max > dsm_min:
                dsm = (dsm - dsm_min) / (dsm_max - dsm_min)
            else:
                dsm = np.zeros_like(dsm)
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)

                image_patches, dsm_patches, pad_hw = pad_batch_to_multiple(image_patches, dsm_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda())
                dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda())

                # Do the inference
                outs = net(image_patches, dsm_patches, mode='Test')
                outs = outs.data.cpu().numpy()
                if pad_hw != (0, 0):
                    outs = outs[:, :, :window_size[0], :window_size[1]]

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
    
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                    np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.00
    for e in range(1, epochs + 1):
        net.train()
        start_time = time.time()
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            outputs = net(data, dsm, mode='Train')
            if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
                output, output_rgb, output_dsm = outputs
                loss1 = loss_calc(output, target, weights)
                pred = output.argmax(dim=1)
                mask_lbl = target.clone()
                mask_lbl[pred == target] = 255
                loss4 = loss_calc(output_rgb, mask_lbl, weights).to(output.device)
                loss5 = loss_calc(output_dsm, mask_lbl, weights).to(output.device)
                aux_w = getattr(unwrap_model(net), "mcrc_aux_weight", 0.01)
                loss = loss1 + aux_w * (loss4 + loss5)
            else:
                output = outputs
                loss = loss_calc(output, target, weights)
            # loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0:
            train_time = time.time()
            print("Training time: {:.3f} seconds".format(train_time - start_time))
            # We validate with the largest possible stride for faster computing
            net.eval()
            MIoU = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            test_time = time.time()
            print("Test time: {:.3f} seconds".format(test_time - train_time))
            if MIoU > MIoU_best:
                if DATASET == 'Vaihingen':
                    os.makedirs('./resultsv', exist_ok=True)
                    torch.save(unwrap_model(net).state_dict(), './resultsv/{}_epoch{}_{}'.format(MODEL_NAME, e, MIoU))
                elif DATASET == 'Potsdam':
                    os.makedirs('./resultsp', exist_ok=True)
                    torch.save(unwrap_model(net).state_dict(), './resultsp/{}_epoch{}_{}'.format(MODEL_NAME, e, MIoU))
                MIoU_best = MIoU
        if scheduler is not None:
            scheduler.step()
    print('MIoU_best: ', MIoU_best)
 
if MODE == 'Train':
    train(net, optimizer, epochs, scheduler, weights=WEIGHTS, save_epoch=save_epoch)

elif MODE == 'Test':
    args = cfg.parse_args()
    weights_path = getattr(args, "weights", None)
    if not weights_path or str(weights_path) == "0":
        raise ValueError("Test mode requires -weights to point to a trained checkpoint.")
    net.load_state_dict(torch.load(weights_path), strict=False)
    net.eval()
    MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
    print("MIoU: ", MIoU)
    for p, id_ in zip(all_preds, test_ids):
        img = convert_to_color(p)
        if DATASET == 'Vaihingen':
            io.imsave('./resultsv/inference_{}_tile_{}.png'.format(MODEL_NAME, id_), img)
        elif DATASET == 'Potsdam':
            io.imsave('./resultsp/inference_{}_tile_{}.png'.format(MODEL_NAME, id_), img)
