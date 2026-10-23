import copy
import os
import random
from typing import Any

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.data_utils import read_client_data, load_split


class SliceDataset(Dataset):
    """Decodes the slice PNGs inside DataLoader worker processes, so the next batches are read, resized and
    pinned while the GPU trains on the current one:
    image -> RGB (2.5D / multi-window), resized to img_size, ToTensor in [0,1]; mask -> uint8 label map,
    nearest-resized to mask_size, int64 (1, H, W). Items: (image_png, mask_png, organ, cls_label); returns (x, y, organ, case id = folder of the slice)."""
    def __init__(self, items, img_size, mask_size):
        self.items, self.mask_size = items, mask_size
        self.tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, msk, organ, cls = self.items[i]
        with Image.open(img) as image, Image.open(msk) as mask:
            x = self.tf(image.convert('RGB'))
            y = mask.convert('L').resize((self.mask_size, self.mask_size), Image.NEAREST)
            y = torch.from_numpy(np.array(y, dtype=np.int64))[None]
        return x, y, str(organ), os.path.basename(os.path.dirname(img))      # case id (patient folder)

class Client(object):

    def __init__(self, args, id, dataset_id, shared_model, **kwargs):
        self.args = args
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.dataset_id = dataset_id
        self.current_round = None

        # full train split: slices (per-epoch slice budget) and volumes (aggregation weight)
        _train = load_split(self.dataset, self.id, self.dataset_id, 'train')
        self.train_size = len(_train)                                           # training slices
        # training volumes (series): slice files are <volume>_<zzzz>_img.png -> aggregation weight of every method
        self.n_volumes = len({os.path.basename(str(it[0])).rsplit("_", 2)[0] for it in _train})
        self.model: Any = None

        if shared_model is not None:
            self.model = copy.deepcopy(shared_model)

        self.train_data = None
        self.test_data = None
        self.val_data = None

        self.batch_size = args.batch_size
        self.local_steps = args.local_steps
        # MedSAM input in [0,1] at img_size; masks are label maps at the same size (logits are upsampled to it)
        self.img_size = getattr(args, "img_size", 512)
        self.mask_size = self.img_size

        self.set_seed(32)

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # data items are (image_path, mask_path, organ_name, cls_label); one organ per client
    def load_train_data(self, batch_size=None, shuffle=True, epoch: Any = 0):
        """A new random subset of `slices_per_epoch` training slices every round and local epoch."""
        batch_size = batch_size or self.batch_size
        max_items = getattr(self.args, "slices_per_epoch", 0) or None
        self.train_data = read_client_data(self.dataset, self.current_round, self.id, self.dataset_id,
                                           data_split='train', epoch=epoch, max_items=max_items)
        batch_size = min(batch_size, len(self.train_data))
        return self._loader(self.train_data, batch_size, shuffle=shuffle, drop_last=True)

    def load_test_data(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        if self.test_data is None:
            self.test_data = read_client_data(self.dataset, self.current_round, self.id, self.dataset_id,
                                              data_split='test')
        batch_size = min(batch_size, len(self.test_data))
        return self._loader(self.test_data, batch_size, shuffle=False, drop_last=False)

    def load_val_data(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        if self.val_data is None:
            self.val_data = read_client_data(self.dataset, "val", self.id, self.dataset_id, data_split='val')
        batch_size = min(batch_size, len(self.val_data))
        return self._loader(self.val_data, batch_size, shuffle=False, drop_last=False)

    def _loader(self, items, batch_size, shuffle, drop_last):
        """Batches of decoded tensors; images are loaded by `--num_workers` worker processes (0 = in the main
        process). The sampling order comes from the main process."""
        nw = int(getattr(self.args, "num_workers", 8))
        kw: dict = dict(num_workers=nw, pin_memory=str(self.device).startswith("cuda"))
        if nw > 0:
            kw["prefetch_factor"] = 4
        return DataLoader(SliceDataset(items, self.img_size, self.mask_size), batch_size,
                          shuffle=shuffle, drop_last=drop_last, **kw)

    def get_eval_model(self, temp_model=None):
        return self.model
