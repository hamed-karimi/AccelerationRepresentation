import argparse
import gc
import json
import os
from types import SimpleNamespace

import nibabel as nib
import nibabel.processing as nibp
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from himalaya.ridge import RidgeCV
from nilearn.glm.first_level import run_glm
from nipy.modalities.fmri.hrf import spmt
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def calculate_column_correlations(A, B):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have the same dimensions")
    
    R = len(A)  # Number of rows
    C = len(A[0])  # Number of columns
    
    def column_mean(matrix, col_index):
        return sum(row[col_index] for row in matrix) / R
    
    def column_std(matrix, col_index, mean):
        variance = sum((row[col_index] - mean)**2 for row in matrix) / R
        return variance**0.5
    
    correlations = []
    
    for col in range(C):
        mean_A = column_mean(A, col)
        mean_B = column_mean(B, col)
        
        std_A = column_std(A, col, mean_A)
        std_B = column_std(B, col, mean_B)
        
        covariance = sum((A[row][col] - mean_A) * (B[row][col] - mean_B) for row in range(R)) / R
        
        if std_A * std_B == 0:
            correlation = 0 
        else:
            correlation = covariance / (std_A * std_B)
        
        correlations.append(correlation)
    
    return correlations


def get_bio_data(subject, run):
    func = nib.load('../forrest_study_fmri/sub-{0:02d}_complete/func/compcorr/func_run-{1:02d}.nii.gz'.format(subject, run))
    gm = nib.load('../forrest_study_fmri/sub-{0:02d}_complete/anat/sub-{1:02d}_T1w_class-GM_probtissue.nii.gz'.format(subject, subject))
    gm_func_size = nibp.resample_from_to(gm, func.slicer[:, :, :, 0])
    return func, gm_func_size

def get_func_masks(func, gm_func_size):
    gm_mask = (gm_func_size.get_fdata() > .1)
    func_fdata = func.get_fdata()
    func_std = func_fdata.std(axis=3)
    bb_mask = (func_std > 0.0)
    bb_gm_mask = np.logical_and(gm_mask, bb_mask)
    return func_fdata, bb_gm_mask

def get_slice_to_region(all_bb_gm_mask):
    slice_to_region = {}
    cc = 0
    for i in range(all_bb_gm_mask.shape[0]):
        for j in range(all_bb_gm_mask.shape[1]):
            for k in range(all_bb_gm_mask.shape[2]):
                if all_bb_gm_mask[i, j, k]:
                    slice_to_region[cc] = [i, j, k]
                    cc += 1
    return slice_to_region


def get_hrf(t1, t2, fps): #fps = 25/5
    hrf_t1_t2 = spmt(np.arange(t1, t2, 1 / fps))
    return hrf_t1_t2


def get_segment_activation(model_layers_dir, seg, layer, batch_split_ind, sampling_rate=5): # -> (time, features, ...)
    layers_dir = '{0}/seg_{1}'.format(model_layers_dir, seg)
    layer_batch_names = [name for name in os.listdir(layers_dir) if '.pt' in name]
    
    layer_batch_path = [os.path.join(layers_dir, name) for name in layer_batch_names]
    layer_activation = []

    for batch_name in layer_batch_path:
        at_batch = torch.load(batch_name, mmap=True, weights_only=True) #.T
        samples = range(0, at_batch.shape[0], sampling_rate)
        layer_activation.append(at_batch[samples])
        del at_batch

    layer_activation = torch.concat(layer_activation, dim=0)
    gc.collect()
    print('memory usage: ', psutil.virtual_memory().used / 1000000000)    
    
    return layer_activation

def layer_convolution(layer_activation):
    convolved = np.zeros(layer_activation.shape, dtype=np.float32)
    hrf = get_hrf(0, 20, 25/5)
    for n in range(layer_activation.shape[0]):# Convolution -> speareds activation to time+99 
        convolved[n, :] = np.convolve(hrf, layer_activation[n, :])[:layer_activation.shape[1]]
        if n % 100000 == 0:
            print('memory usage in conv: ', psutil.virtual_memory().used / 1000000000)
            gc.collect()
    gc.collect()
    return convolved

def get_frame2_loc(batched_flow1, batched_flow2):
    B, C, W, H = batched_flow1.shape
    grid_x = torch.linspace(0, W-1, W, dtype=int)
    grid_y = torch.linspace(0, H-1, H, dtype=int)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')  # shape: [H, W]
    
    # Stack to get base grid, shape [H, W, 2]
    frame1_loc = torch.stack((grid_x, grid_y), dim=0)  # (H,W,2)
    frame1_loc = frame1_loc.unsqueeze(0).repeat(B,1,1,1)  # (B,H,W,2)
    
    frame2_loc = torch.zeros_like(frame1_loc)
    frame2_loc[:, 0, :, :] = frame1_loc[:, 0, :, :] + batched_flow1[:, 0, :, :] # + torch.round(input=batched_flow1[:, 0, :, :], decimals=0)
    frame2_loc[:, 1, :, :] = frame1_loc[:, 1, :, :] + batched_flow1[:, 1, :, :] # + torch.round(input=batched_flow1[:, 1, :, :], decimals=0)

    return frame2_loc, batched_flow1, batched_flow2

def compute_flow_difference(batched_flow1, batched_flow2, frame2_loc):
    batch_size, _, H, W = batched_flow1.shape
    new_x = frame2_loc[:, 0, :, :]
    new_y = frame2_loc[:, 1, :, :]
    
    # Normalize coordinates to [-1, 1] for grid_sample (x: W dimension, y: H dimension)
    # Normalize new_x from [0, W-1] -> [-1, 1]
    new_x_norm = 2.0 * new_x / (W - 1) - 1.0
    # Normalize new_y from [0, H-1] -> [-1, 1]
    new_y_norm = 2.0 * new_y / (H - 1) - 1.0
    
    grid = torch.stack((new_x_norm, new_y_norm), dim=-1)  # last dimension = 2, (x, y)
    
    batched_flow2 = batched_flow2.float()
    grid = grid.float()
    
    flow2_sampled = F.grid_sample(batched_flow2, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    valid_mask = (new_x >= 0) & (new_x <= (W-1)) & (new_y >= 0) & (new_y <= (H-1))
    valid_mask = valid_mask.unsqueeze(1).float()  # shape (batch, 1, H, W)
    
    flow_diff = (flow2_sampled - batched_flow1) * valid_mask
    
    return flow_diff

def get_train_pcs(train_activation, n_components=100):
    pca = PCA(n_components=n_components)
    transformed_activation = pca.fit_transform(train_activation)
    return transformed_activation, pca

def get_segment_t_map(fmri_pred, transformed_fmri_scaler):
    n_voxels = fmri_pred.shape[1]
    time = fmri_pred.shape[0]
    t_map = np.zeros((n_voxels,))
    effect_map = np.zeros((n_voxels, ))
    var_map = np.zeros((n_voxels, ))
    Y = transformed_fmri_scaler # time x voxel
    for loc in range(n_voxels):
        X = np.column_stack([fmri_pred[:, loc:loc+1], np.ones(time)])
        y = Y[:, loc:loc+1]
        labels_v, results_v = run_glm(y, X, n_jobs=-1, noise_model='ar1')
        label = labels_v[0]
        t_map[loc] = results_v[label].t()[0, 0]
        effect_map[loc] = results_v[label].Tcontrast(np.array([1, 0])).effect.astype('float32')
        var_map[loc] = (results_v[label].Tcontrast(np.array([1, 0])).sd ** 2).astype('float32')
    return t_map, effect_map, var_map


ALLOWED_MODULES = ('velocity', 'acceleration')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Fit ridge encoding model for velocity or acceleration.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'examples:\n'
            '  python encoding_model.py 1 --module velocity\n'
            '  python encoding_model.py 1 --module acceleration'
        ),
    )
    parser.add_argument(
        'subject',
        type=int,
        help='Subject ID (e.g. 1)',
    )
    parser.add_argument(
        '--module',
        required=True,
        choices=ALLOWED_MODULES,
        help='Motion module to encode (velocity or acceleration)',
    )
    args = parser.parse_args(argv)
    return args.subject, args.module


def main():
    model_name = 'RAFT' #sys.argv[1] 
    mode = 'FlyingThings3D-trained-cropped' #sys.argv[2] 
    subject, module_save_name = parse_args()
    ref_subject = 1
    features_dir = os.path.join('../features', model_name, mode)
    params_dir = '../model/Parameters.json'
    with open(params_dir, 'r') as json_file:
            params = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    model_modules = params.MODULES
    analysis = 'GLMResult'
    do_convolution = True
    first_frame = 25
    split_index = params.SPLIT_INDEX
    per_segment = True
    if per_segment:
        res_dir = os.path.join('../results/EncodingModel', f'subject-{subject:02d}', module_save_name, 'segments', analysis)
    else:
        res_dir = os.path.join('../results/EncodingModel', f'subject-{subject:02d}', module_save_name)
        
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # DNN
    all_seg_model_activation_list = []
    for seg in range(8):
        module = model_modules[0]
        layer_activation = get_segment_activation(model_layers_dir=features_dir, 
                                                  seg=seg, 
                                                  layer=module.replace('.', '_'),
                                                  batch_split_ind=split_index,
                                                  sampling_rate=1)

        fmri_time = nib.load('../forrest_study_fmri/sub-{0:02d}_complete/func/compcorr/func_run-{1:02d}.nii.gz'.format(ref_subject, seg+1)).shape[3]
        frames_trigger = (np.array([50] * fmri_time).cumsum() - (50 - first_frame)) // (25//5)

        if module_save_name == 'acceleration':
            frame2_loc, batched_flow1, batched_flow2 = get_frame2_loc(
                layer_activation[:-1], layer_activation[1:])
            flow_diff = compute_flow_difference(batched_flow1, batched_flow2, frame2_loc)
            print('flow_diff done')
            sampling_rate = 5
            samples = range(0, flow_diff.shape[0], sampling_rate)
            features = flow_diff[samples].flatten(start_dim=1, end_dim=-1)
        else:  # velocity
            sampling_rate = 5
            samples = range(0, layer_activation.shape[0], sampling_rate)
            features = layer_activation[samples].flatten(start_dim=1, end_dim=-1)

        convolved = layer_convolution(features.T) # convolved.shape = features x time
        convolved_corrected_time = convolved[:, frames_trigger]

        all_seg_model_activation_list.append(convolved_corrected_time)

    all_seg_model_activation = np.concat(all_seg_model_activation_list, axis=1)

    # fMRI
    func, gm_func_size = get_bio_data(subject, 1)
    all_bb_gm_mask = np.ones(gm_func_size.shape, dtype=bool)
    subject_all_runs_fmri_list = []
    for run in range(1, 8+1):
        func, gm_func_size = get_bio_data(subject, run)
        func_fdata, bb_gm_mask = get_func_masks(func, gm_func_size)
        all_bb_gm_mask = all_bb_gm_mask & bb_gm_mask
        subject_all_runs_fmri_list.append(func_fdata)    
    slice_to_region = get_slice_to_region(all_bb_gm_mask)


    seg_t_map = []

    t_map = np.zeros((8,
                      all_bb_gm_mask.shape[0],
                      all_bb_gm_mask.shape[1],
                      all_bb_gm_mask.shape[2]))
    
    for test_seg in range(8):
        test_run = test_seg + 1
        train_activation_list = [all_seg_model_activation_list[s] for s in range(8) if s != test_seg]
        train_activation = np.concat(train_activation_list, axis=1) # features x time
        test_activation = all_seg_model_activation_list[test_seg]
        
        train_fmri_list = []
        for s in range(8):
            if s!= test_seg:
                train_fmri_list.append(subject_all_runs_fmri_list[s][all_bb_gm_mask, :])
        train_fmri = np.concat(train_fmri_list, axis=1) # voxels x time
        test_fmri = subject_all_runs_fmri_list[test_seg][all_bb_gm_mask, :]

        model_scaler = StandardScaler(with_mean=True, with_std=False)
        fmri_scaler = StandardScaler(with_mean=True, with_std=False)
        
        train_activation_scaled = model_scaler.fit_transform(train_activation.T)
        transformed_train_activation, pca = get_train_pcs(train_activation=train_activation_scaled, n_components=100) # time x n_pcs
        
        test_activation_scaled = model_scaler.transform(test_activation.T)
        transformed_test_activation = pca.transform(test_activation_scaled) # time x n_pcs

        alphas = np.logspace(5, 20, 80)
        model = RidgeCV(alphas=alphas, 
                        fit_intercept=True, 
                        cv=10, 
                        Y_in_cpu=False, 
                        force_cpu=False)
        
        model.fit(transformed_train_activation, 
                  fmri_scaler.fit_transform(train_fmri.T))
        
        fmri_pred = model.predict(transformed_test_activation)
        t, _, _ = get_segment_t_map(fmri_pred,
                                    fmri_scaler.transform(test_fmri.T))
        seg_t_map.append(t)

        np.save(os.path.join(res_dir, f'seg_{test_seg}_model_coef.npy'), model.coef_)
        
        print(test_seg, 'done')


    t_np = np.stack(seg_t_map, axis=0)

    for r in slice_to_region:
        t_map[:,
        slice_to_region[r][0],
        slice_to_region[r][1],
        slice_to_region[r][2]] = t_np[:, r]

    np.save(os.path.join(res_dir, 't_map.npy'), t_map)


if __name__ == '__main__':
    main()
