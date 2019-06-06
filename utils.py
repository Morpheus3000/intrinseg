import datetime
import sys
import torch
import os
import numpy as np
from PIL import Image

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_image(image_data):
    if image_data.shape[0] == 1:
        image = image_data[0]
        image = (image * 255.0).astype(np.uint8)
        image = image.transpose((1, 2, 0))
        image = np.squeeze(image)
    else:
        image = image_data
    pil_image = Image.fromarray(image)
    return pil_image

def write_image(path, image_data, file_name, type):
    path = os.path.join(path, 'images', type)
    pil_image = create_image(image_data)
    output_path = os.path.join(path, file_name)
    output_folder = os.path.dirname(output_path)
    create_folder(output_folder)
    pil_image.save(output_path)


class network:
    @staticmethod
    def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, lr_decay_iter=1,
                          power=0.9):
        """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

        """
        if iter % lr_decay_iter or iter > max_iter:
            return optimizer

        lr = init_lr * (1 - iter / max_iter) ** power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    @staticmethod
    def save_checkpoint(path, net, epoch):
        # Saves in path/checkpoints/{epoch}.checkpoint
        path = os.path.join(path, "checkpoints")
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, "epoch_%d.checkpoint" % epoch)

        torch.save(net.state_dict(), path)

    @staticmethod
    def load_checkpoint(path):
        # Loads last checkpoint in path/checkpoints
        path = os.path.join(path, "checkpoints")
        print(path)
        print('Loaded: ', path)
        if os.path.exists(path):
            checkpoints_available = [f.name for f in os.scandir(path) if f.is_file() and f.name.endswith(".checkpoint")]
            epochs = [int(name.replace("epoch_", "").replace(".checkpoint", "")) for name in checkpoints_available]
            latest = os.path.join(path, "epoch_%d.checkpoint" % max(epochs))
            output("Loading checkpoint from epoch %s" % latest)
            print(latest)
            return torch.load(latest),  max(epochs)
        else:
            return None

    @staticmethod
    def load_checkpoint_state(path, net, epoch=None):
        # Loads last checkpoint in path/checkpoints
        path = os.path.join(path, "checkpoints")
        if os.path.exists(path):
            if epoch is None:
                checkpoints_available = [f.name for f in os.scandir(path) if f.is_file() and f.name.endswith(".checkpoint")]
                epochs = [int(name.replace("epoch_", "").replace(".checkpoint", "")) for name in checkpoints_available]
                epoch = max(epochs)

            latest = os.path.join(path, "epoch_%d.checkpoint" % epoch)
            output("Loading checkpoint from epoch %s" % latest)
            net.load_state_dict(torch.load(latest))
            return epoch
        else:
            print(path)
            print("Checkpoint not found")
            return None

    @staticmethod
    def load_finetune_state(path, net, epoch):
        # Loads last checkpoint in path/checkpoints
        path = os.path.join(path, "checkpoints")
        if os.path.exists(path):
            latest = os.path.join(path, "epoch_%d.checkpoint" % epoch)
            output("Loading checkpoint from epoch %s" % latest)

            pretrained_dict = torch.load(latest)
            net_dict = net.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k == "output.2.3.weight" and not k == "output.2.3.bias" and not k == "output.2.4.weight" and not k == "output.2.4.bias" and not k == 'output.2.4.running_mean' and not k == 'output.2.4.running_var'}
            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict)
            return epoch
        else:
            print(path)
            print(None)
            return None

    @staticmethod
    def save_weights(path, net):
        # Saves model in path/model.weights
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, "model.weights")
        torch.save(net.state_dict(), path)

    @staticmethod
    def load_weights(path, net):
        # Loads model path/model.weights
        path = os.path.join(path, "model.weights")
        print('Loaded: ', path)
        if os.path.isfile(path):
            net.load_state_dict(torch.load(path))
            return True
        else:
            return False

class ExperimentType:
    # names: intrinsics, segmentation, combined
    def __init__(self, name):
        self.intrinsics = name == 'intrinsics'
        self.segmentation = name == 'segmentation'
        self.combined = name == 'combined'
        self.rbgsegs = name == 'rgbsegs'
        self.trimtune = name == 'trimtune'


class Cuda:
    def __init__(self, device):
        self.enabled = device is not -1
        self.device = device

class TrimBot:
    label_names = [
        "void",
        "ground",
        "grass",
        "dirt",
        "gravel",
        "woodchip",
        "pavement",
        "box",
        "topiary",
        "rose",
        "tree",
        "fence",
        "step",
        "flowerpot",
        "stone",
        "sky",
    ]

    @staticmethod
    def color_labels(labels):
        color_map = [
            [0, 0, 0],  # Unknown 0
            [0, 0.8, 0],  # Grass 1
            [0.3, 0.5, 0],  # Ground 2 (dirt)
            [0.3, 0.5, 0],  # Ground 3 (pebbles)
            [0.3, 0.5, 0],  # Ground 4 (woodchip)
            [0.7, 0.8, 0.9],  # Pavement 5
            [0.5, 0.5, 0],  # Hedge 6
            [0, 0.7, 0.7],  # Topiary 7
            [0.9, 0, 0],  # Rose 8
            [0.2, 0.2, 0.9],  # Obstacle 9
            [0.3, 0.7, 0.1],  # Tree 10
            [0.1, 0.1, 0.1],  # Background 11
        ]
        color_map = np.array(color_map)
        color_map = np.array(color_map * 255, dtype=np.uint8)

        labels = labels.squeeze()
        labels = labels.astype(np.uint8)
        image = np.empty((labels.flatten().shape[0], 3))
        image[:] = np.array([color_map[TrimBot.garden_map_reverse()[label]].transpose() for label in labels.flatten()])

        image = image.reshape((labels.shape[0], labels.shape[1], 3))

        return image.astype(np.uint8)

    @staticmethod
    def garden_map():
        return {
            0: 0,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 11,
            10: 10,
            11: 0,
        }

    @staticmethod
    def garden_map_reverse():
        return {
            0: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            8: 7,
            9: 8,
            11: 9,
            10: 10,
            12: 0,
            13: 0,
            14: 0,
            15: 0
        }


class Garden:
    label_names = [
        "void",
        "ground",
        "grass",
        "dirt",
        "gravel",
        "woodchip",
        "pavement",
        "box",
        "topiary",
        "rose",
        "tree",
        "fence",
        "step",
        "flowerpot",
        "stone",
        "sky",
    ]

    @staticmethod
    def color_labels(labels):
        color_map = [
            [0, 0, 0],  # void
            [0, 76, 153],  # ground
            [0, 153, 0],  # grass
            [102, 51, 0],  # dirt
            [0, 204, 204],  # gravel
            [204, 102, 0],  # woodchip
            [102, 102, 255],  # pavement
            [255, 204, 153],  # box
            [153, 255, 153],  # topiary
            [255, 153, 153],  # rose
            [153, 153, 0],  # tree
            [51, 51, 153],  # fence
            [64, 64, 64],  # step
            [204, 0, 102],  # flowerpot
            [51, 0, 25],  # stone
            [128, 128, 128],  # sky
        ]

        labels = labels.squeeze()
        labels = labels.astype(np.uint8)
        image = np.empty((labels.flatten().shape[0], 3))
        image[:] = np.array([np.array(color_map[label]).transpose() for label in labels.flatten()])

        image = image.reshape((labels.shape[0], labels.shape[1], 3))

        return image.astype(np.uint8)

    @staticmethod
    def get_label(color):
        label_map = {
            (0, 0, 0): 0,  # void
            (0, 76, 153): 1,  # ground
            (0, 153, 0): 2,  # grass
            (102, 51, 0): 3,  # dirt
            (0, 204, 204): 4,  # gravel
            (204, 102, 0): 5,  # woodchip
            (102, 102, 255): 6,  # pavement
            (255, 204, 153): 7,  # box
            (153, 255, 153): 8,  # topiary
            (255, 153, 153): 9,  # rose
            (153, 153, 0): 10,  # tree
            (51, 51, 153): 11,  # fence
            (64, 64, 64): 12,  # step
            (204, 0, 102): 13,  # flowerpot
            (51, 0, 25): 14,  # stone
            (128, 128, 128): 15,  # sky
        }

        return label_map.get(tuple(color))

    @staticmethod
    def label_map(image):
        image = image.squeeze()
        image = image.astype(np.uint8)
        flat_image = image.reshape(-1, image.shape[-1])

        # labels = np.empty((flat_image.shape[0]), dtype=np.uint8)
        labels = np.apply_along_axis(Garden.get_label, 1, flat_image)
        # labels[:] = [label_map[tuple(color)] for color in flat_image]
        labels = labels.reshape(image.shape[0:2])
        return labels

