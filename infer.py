import argparse

import os
import torch
from torch.autograd import Variable
from PIL import Image
import time
import glob
from tqdm import tqdm

from direct_intrinsics_sn import DirectIntrinsicsSN
from utils import Cuda, ExperimentType, network, write_image, Garden

import numpy as np



def set_experiment(experiment_name):
    output_folder = "experiment/%s" % experiment_name

    experiment_keys = experiment_name.split("-")

    if "intrinsics" in experiment_keys:
        exp_type = ExperimentType('intrinsics')
    if "segmentation" in experiment_keys:
        exp_type = ExperimentType('segmentation')
    if "combined" in experiment_keys:
        exp_type = ExperimentType('combined')

    print(experiment_keys)
    normalize = True
    grayscale_shading = "gray" in experiment_keys
    balance_loss = "loss_balance" in experiment_keys
    learn_upsample = "up" in experiment_keys
    pooling = "pooling" in experiment_keys
    recon_loss = "recon" in experiment_keys

    if exp_type.intrinsics:
        net_in = ['rgb']
        net_out = ['albedo', 'shading']
    elif exp_type.segmentation:
        net_in = ['rgb']
        net_out = ['segmentation']
    elif exp_type.combined:
        net_in = ['rgb']
        net_out = ['albedo', 'shading', 'segmentation']

    return output_folder, exp_type, net_in, net_out, normalize, grayscale_shading, balance_loss, learn_upsample, pooling, recon_loss


def main(filenames, name, gpu, model_loc, result_loc):
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
    cuda = Cuda(0)

    net_in = ['rgb']
    net_out = ['albedo', 'shading', 'segmentation']
    normalize = True
    net = DirectIntrinsicsSN(3, ['color', 'color', 'class'])

    # The following points to where the weight file is kept. Keep it above
    # the experiment folder. The experiment folder is attached along with the experiment,
    # based on the name
    output_path_root = model_loc
    output_path = os.path.join(output_path_root, 'experiment', name)

    # Change the following to the place where the image should be written
    results_path = os.path.join(result_loc, name)
    #os.makedirs(results_path, exist_ok=True)

    if not network.load_weights(output_path, net):
        # We load the model here. You can modify the os.path.join arguments to
        # reflect to your desired folder structure. We use an extra folder for
        # the checkpoints to keep it organized.
        net.load_state_dict(torch.load(os.path.join(output_path, 'checkpoints',
                                                    'final.checkpoint')))
    if cuda.enabled:
        net.cuda(device=cuda.device)

    net.eval()

    if os.path.isdir(filenames):
        print('Found directory. Scanning for png images...')
        files = glob.glob(os.path.join(filenames, '*.png'))
        print('Done! Found %d files' % len(files))
        for filename in tqdm(files):
            im = Image.open(filename)
            in_ = im.resize((352, 480), Image.ANTIALIAS)
            in_ = np.array(in_, dtype=np.int64)
            in_ = in_.astype(np.float32)

            in_[np.isnan(in_)] = 0
            in_ = in_.transpose((2, 0, 1))
            if normalize:
                in_ = (in_ * 255 / np.max(in_)).astype('uint8')
                in_ = (in_ / 255.0).astype(np.float32)

            in_ = np.expand_dims(in_, axis=0)
            rgb = torch.from_numpy(in_)
            if cuda.enabled:
                rgb = Variable(rgb).cuda(device=cuda.device)
            else:
                rgb = Variable(rgb)
            albedo_out, shading_out, segmentation_out = net(rgb)
            filename = filename.split('/')[-1]
            write_image(results_path, albedo_out.detach().cpu().numpy(), filename, 'albedo')
            write_image(results_path, shading_out.detach().cpu().numpy(), filename, 'shading')
            _, segmentation_pred = torch.max(segmentation_out.data, 1)
            write_image(results_path,
                        Garden.color_labels(segmentation_pred.cpu().numpy()), filename, 'segmentation')
    else:
        print('Target is a single image.')
        filename = filenames
        im = Image.open(filename)
        in_ = im.resize((352, 480), Image.ANTIALIAS)
        in_ = np.array(in_, dtype=np.int64)
        in_ = in_.astype(np.float32)

        in_[np.isnan(in_)] = 0
        in_ = in_.transpose((2, 0, 1))
        if normalize:
            in_ = (in_ * 255 / np.max(in_)).astype('uint8')
            in_ = (in_ / 255.0).astype(np.float32)

        in_ = np.expand_dims(in_, axis=0)
        rgb = torch.from_numpy(in_)
        if cuda.enabled:
            rgb = Variable(rgb).cuda(device=cuda.device)
        else:
            rgb = Variable(rgb)
        print('Data loaded! Beginning inference')
        a = time.time()
        albedo_out, shading_out, segmentation_out = net(rgb)
        print('Inference completed in %.2f seconds' % (time.time() - a))
        print('Writing results')
        write_image(results_path, albedo_out.detach().cpu().numpy(), filename, 'albedo')
        write_image(results_path, shading_out.detach().cpu().numpy(), filename, 'shading')
        _, segmentation_pred = torch.max(segmentation_out.data, 1)
        write_image(results_path,
                    Garden.color_labels(segmentation_pred.cpu().numpy()), filename, 'segmentation')
    print('Done! Shutting down...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=False, default=0)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--results', type=str, required=False,
                        default='./results')
    args = parser.parse_args()
    args.results = os.path.join(args.results, 'disn/')

    main(args.file, args.name, args.gpu, args.model, args.results)
