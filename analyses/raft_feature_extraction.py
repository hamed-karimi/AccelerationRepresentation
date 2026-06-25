import argparse
import os
import sys

sys.path.append('../RAFT/core')

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from raft import RAFT
from raft_dataloader import FG_Dataset
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler


def get_segment_dataloader(frames_path, batch_size, start_frame, segment, window):
    dataset = FG_Dataset(frames_path, segment, window=window)

    end_frame = len(dataset)
    print(start_frame, end_frame)
    batch_sampler = BatchSampler(
        SequentialSampler(range(start_frame, end_frame)),
        batch_size=batch_size,
        drop_last=False,
    )
    data_generator = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        pin_memory=True,
    )

    return data_generator, len(dataset)


def get_result_dir(root, mode, segment):
    result_dir = os.path.join(root, mode, 'seg_{0}'.format(segment))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def parse_raft_arguments(args):
    opt_cmd = {}
    for arg in args:
        assert arg.startswith("--")
        if "=" not in arg[2:]:
            key_str, value = (arg[2:-1], "false") if arg[-1] == "!" else (arg[2:], "true")
        else:
            key_str, value = arg[2:].split("=")
        keys_sub = key_str.split(".")
        opt_sub = opt_cmd
        for k in keys_sub[:-1]:
            if k not in opt_sub:
                opt_sub[k] = {}
            opt_sub = opt_sub[k]
        assert keys_sub[-1] not in opt_sub, keys_sub[-1]
        opt_sub[keys_sub[-1]] = yaml.safe_load(value)
    return edict(opt_cmd)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Extract RAFT optical-flow features for one movie segment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'example:\n'
            '  python raft_feature_extraction.py /path/to/frames --segment 0'
        ),
    )
    parser.add_argument(
        'frames_path',
        help='Path to stimulus frames directory (contains seg_0 ... seg_7)',
    )
    parser.add_argument(
        '--segment',
        type=int,
        required=True,
        choices=range(8),
        help='Movie segment index (0-7)',
    )
    args = parser.parse_args(argv)
    if not os.path.isdir(args.frames_path):
        parser.error(f'frames_path does not exist or is not a directory: {args.frames_path!r}')
    segment_dir = os.path.join(args.frames_path, f'seg_{args.segment}')
    if not os.path.isdir(segment_dir):
        parser.error(f'segment directory not found: {segment_dir!r}')
    return args.frames_path, args.segment


def main():
    frames_path, seg = parse_args()

    raft_args = [
        '--model=../RAFT/models/raft-things.pth',
        '--path=demo-frames',
        '--small=False',
        '--mixed_precision',
    ]
    args_cmd = parse_raft_arguments(raft_args)
    model = torch.nn.DataParallel(RAFT(args_cmd))
    model.load_state_dict(torch.load(args_cmd.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {0}: '.format(device))
    model.to(device)
    model.eval()

    model_name = 'RAFT'
    mode = 'FlyingThings3D-trained-cropped'
    window = 2
    batch_size = 200
    start_frame = 0

    data_generator, n_dataset = get_segment_dataloader(
        frames_path=frames_path,
        batch_size=batch_size,
        start_frame=start_frame,
        segment=seg,
        window=window,
    )
    flow_tensors = torch.empty((n_dataset, 2, 224, 224))
    result_dir = get_result_dir('../features/{0}'.format(model_name), mode, seg)
    print(result_dir)

    for ii, image_batch in enumerate(data_generator):
        with torch.no_grad():
            image_batch_gpu = image_batch.to(device)
            image1_batch_gpu = image_batch_gpu[:, :, 0, :, :]
            image2_batch_gpu = image_batch_gpu[:, :, 1, :, :]
            _, flow = model.module(
                image1_batch_gpu, image2_batch_gpu, iters=12, test_mode=True)

        flow_tensors[ii * batch_size:ii * batch_size + image_batch.shape[0], :, :, :] = flow.cpu()
        print(ii * batch_size, ii * batch_size + image_batch.shape[0] - 1)

    torch.save(flow_tensors, os.path.join(result_dir, 'flow.pt'))


if __name__ == '__main__':
    main()
