import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import itertools
from torchvision.utils import make_grid
from PIL import Image
from skimage import io
import os

# Reproducibility
SEED = 42

def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Parameters
## SwinFusion
WINDOW_SIZE = (256, 256) # Patch size
PATCH_MULTIPLE = 14

STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "/data1/lihaocheng/deeplearning/dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
# Keep total samples per epoch fixed; adjust BATCH_SIZE for speed without changing data amount.
BASE_BATCH_SIZE = 24
ITERATIONS_PER_EPOCH = 1000
SAMPLES_PER_EPOCH = BASE_BATCH_SIZE * ITERATIONS_PER_EPOCH
BATCH_SIZE = BASE_BATCH_SIZE

# BATCH_SIZE = 4 # For backbone ViT-Huge
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(3, 1, 1)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(3, 1, 1)

# Used for logging/checkpoint naming only.
MODEL_NAME = 'MMNet'
MODE = None

# DATASET = 'Vaihingen'
DATASET = 'Potsdam'
ENCODER = None


try:
    import Model.cfg as cfg
    _args = cfg.parse_args()
    ENCODER = str(getattr(_args, "encoder", "dinov2_vitl14")).strip()
    MODE = str(getattr(_args, "mode", "Train")).strip().capitalize()
    exp_name = str(getattr(_args, "exp_name", "")).strip()
    if exp_name:
        MODEL_NAME = f"{MODEL_NAME}_{exp_name}"
except Exception:
    if ENCODER is None:
        ENCODER = "dinov2_vitl14"
    if MODE is None:
        MODE = "Train"
USE_IMAGENET_NORM = str(ENCODER).startswith("dinov2")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CACHE = True # Store the dataset in-memory

if DATASET == 'Vaihingen':
    train_ids = ['1', '3', '5', '7', '11', '13', '15', '17', '21', '23', '26', '28', '30', '32', '34', '37']
    test_ids = ['2', '4', '6', '8', '10', '12', '14', '16', '20', '22','24','27','29','31','33','35','38']
    Stride_Size = 32
    STRIDE = 16 # Stride for testing
    # Stride_Size = 64
    # STRIDE = 32 # Stride for testing
    epochs = 50
    save_epoch = 5
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'label/top_mosaic_09cm_area{}_noBoundary.tif'
    LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
    N_CLASSES = len(LABELS) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    # ISPRS color palette
    palette = {0 : (255, 255, 255), # Impervious surfaces (white)
               1 : (0, 0, 255),     # Buildings (blue)
               2 : (0, 255, 255),   # Low vegetation (cyan)
               3 : (0, 255, 0),     # Trees (green)
               4 : (255, 255, 0),   # Cars (yellow)
               5 : (255, 0, 0),     # Clutter (red)
               6 : (0, 0, 0)}       # Undefined (black)
    invert_palette = {v: k for k, v in palette.items()}
elif DATASET == 'Potsdam':
    train_ids = ['2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11', '4_12', '5_10', '5_11', '5_12', '6_7',
                '6_8', '6_9', '6_10', '6_11', '6_12', '7_7', '7_8', '7_9', '7_11', '7_12']
    test_ids = ['2_13', '2_14', '3_13', '3_14', '4_13', '4_14', '4_15', '5_13', '5_14', '5_15', '6_13', '6_14', '6_15', '7_13']
    Stride_Size = 64
    STRIDE = 32 # Stride for testing
    epochs = 50
    save_epoch = 5
    MAIN_FOLDER = FOLDER + 'Potsdam/'
    DATA_FOLDER = MAIN_FOLDER + 'rgb/top_potsdam_{}_RGB.tif'
    DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_potsdam_{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'labels/top_potsdam_{}_label_noBoundary.tif'
    LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
    N_CLASSES = len(LABELS) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    # ISPRS color palette
    palette = {0 : (255, 255, 255), # Impervious surfaces (white)
               1 : (0, 0, 255),     # Buildings (blue)
               2 : (0, 255, 255),   # Low vegetation (cyan)
               3 : (0, 255, 0),     # Trees (green)
               4 : (255, 255, 0),   # Cars (yellow)
               5 : (255, 0, 0),     # Clutter (red)
               6 : (0, 0, 0)}       # Undefined (black)
    invert_palette = {v: k for k, v in palette.items()}
print(MODEL_NAME + ', ' + MODE + ', ' + DATASET + ', ENCODER: ' + str(ENCODER) + ', WINDOW_SIZE: ', WINDOW_SIZE, 
      ', BATCH_SIZE: ' + str(BATCH_SIZE), ', Stride_Size: ', str(Stride_Size),
      ', epochs: ' + str(epochs), ', save_epoch: ', str(save_epoch),)

def _potsdam_dsm_id(tile_id: str) -> str:
    parts = tile_id.split("_", 1)
    if len(parts) != 2:
        return tile_id
    head, tail = parts
    if len(head) == 1:
        head = f"0{head}"
    if len(tail) == 1:
        tail = f"0{tail}"
    return f"{head}_{tail}"

def normalize_rgb(arr):
    if not USE_IMAGENET_NORM:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 3:
            return (arr - RGB_MEAN) / RGB_STD
        if arr.shape[-1] == 3:
            mean = RGB_MEAN.reshape(1, 1, 3)
            std = RGB_STD.reshape(1, 1, 3)
            return (arr - mean) / std
    if arr.ndim == 4:
        if arr.shape[1] == 3:
            mean = RGB_MEAN.reshape(1, 3, 1, 1)
            std = RGB_STD.reshape(1, 3, 1, 1)
            return (arr - mean) / std
        if arr.shape[-1] == 3:
            mean = RGB_MEAN.reshape(1, 1, 1, 3)
            std = RGB_STD.reshape(1, 1, 1, 3)
            return (arr - mean) / std
    return arr

def match_spatial_shape(arr, ref_shape, pad_mode="edge"):
    """Match H,W of arr to ref_shape (H,W). Crops or pads as needed."""
    ref_h, ref_w = ref_shape[:2]
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    h, w = arr.shape[:2]

    if h > ref_h or w > ref_w:
        arr = arr[:ref_h, :ref_w]
        h, w = arr.shape[:2]

    pad_h = ref_h - h
    pad_w = ref_w - w
    if pad_h > 0 or pad_w > 0:
        pad_h = max(0, pad_h)
        pad_w = max(0, pad_w)
        pad_width = ((0, pad_h), (0, pad_w))
        if arr.ndim == 3:
            pad_width = pad_width + ((0, 0),)
        arr = np.pad(arr, pad_width, mode=pad_mode)
    return arr

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def _pad_with_hw(arr, pad_h, pad_w, value):
    if pad_h == 0 and pad_w == 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-2] = (0, pad_h)
    pad_width[-1] = (0, pad_w)
    return np.pad(arr, pad_width, mode='constant', constant_values=value)

def pad_triplet_to_multiple(data, dsm, label, multiple=PATCH_MULTIPLE):
    h, w = label.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return data, dsm, label, (0, 0)
    data = _pad_with_hw(data, pad_h, pad_w, value=0.0)
    dsm = _pad_with_hw(dsm, pad_h, pad_w, value=0.0)
    label = _pad_with_hw(label, pad_h, pad_w, value=255)
    return data, dsm, label, (pad_h, pad_w)

def pad_batch_to_multiple(images, dsms, multiple=PATCH_MULTIPLE):
    h, w = images.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return images, dsms, (0, 0)
    images = _pad_with_hw(images, pad_h, pad_w, value=0.0)
    dsms = _pad_with_hw(dsms, pad_h, pad_w, value=0.0)
    return images, dsms, (pad_h, pad_w)

def save_img(tensor, name):
    tensor = tensor.cpu() .permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')

class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        if DATASET == 'Potsdam':
            self.dsm_files = [DSM_FOLDER.format(_potsdam_dsm_id(id)) for id in ids]
        else:
            self.dsm_files = [DSM_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        if DATASET == 'Potsdam' or DATASET == 'Vaihingen':
            return SAMPLES_PER_EPOCH
        return None

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            ## Potsdam IRRG
            if DATASET == 'Potsdam':
                ## RGB
                data = io.imread(self.data_files[random_idx])[:, :, :3].transpose((2, 0, 1))
                ## IRRG
                # data = io.imread(self.data_files[random_idx])[:, :, (3, 0, 1, 2)][:, :, :3].transpose((2, 0, 1))
                data = 1 / 255 * np.asarray(data, dtype='float32')
            else:
            ## Vaihingen IRRG
                data = io.imread(self.data_files[random_idx])
                data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            data = normalize_rgb(data)
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[random_idx]
        else:
            # DSM is normalized in [0, 1]
            dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')
            min = np.min(dsm)
            max = np.max(dsm)
            dsm = (dsm - min) / (max - min + 1e-8)
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            invalid = (label < 0) | (label >= N_CLASSES)
            if np.any(invalid):
                label = label.copy()
                label[invalid] = 255
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)
        data_p, dsm_p, label_p, _ = pad_triplet_to_multiple(data_p, dsm_p, label_p)
        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))
        
def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

class CrossEntropy2d_ignore(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        valid = target != self.ignore_label
        if valid.sum() == 0:
            return predict.sum() * 0.0
        loss = F.cross_entropy(
            predict,
            target,
            weight=weight,
            ignore_index=self.ignore_label,
            reduction="mean",
        )
        return loss
    
def loss_calc(pred, label, weights):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d_ignore().cuda()

    return criterion(pred, label, weights)

def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" %(kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:5])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")

    return MIoU
