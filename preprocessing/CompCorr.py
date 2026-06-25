import numpy as np
import nibabel.processing as nibp
import os
import shutil
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn import linear_model
from copy import deepcopy


datapath = '../forrest_study_fmri' 

def get_file_path(subject_num, mode, run=None):
    subject_dir_name = 'sub-{:02d}_complete'.format(subject_num)
    if mode == 'functional':
        func_dir = os.path.join(datapath, subject_dir_name, 'func')
        func = 'sub-{:02d}_ses-movie_task-movie_run-{:d}_bold_space-T1w_preproc.nii.gz'.format(subject_num, run)
        return os.path.join(func_dir, func)
    elif mode == 'gm' or mode == 'wm' or mode == 'csf':
        anat_dir = os.path.join(datapath, subject_dir_name, 'anat')
        img = 'sub-{:02d}_T1w_class-{:s}_probtissue.nii.gz'.format(subject_num, mode.upper())
        return os.path.join(anat_dir, img)


for subject in [1, 2, 3, 4, 5, 9, 10, 14, 15, 16, 17, 18, 19, 20]:
    func = []
    for run in range(1, 9):
        filepath_func = get_file_path(subject, mode='functional', run=run)
        func.append(nib.load(filepath_func))
    
    func = np.array(func)
    filepath_gm = get_file_path(subject, mode='gm')
    gm = nib.load(filepath_gm)
#     print(gm.affine)
    
    filepath_wm = get_file_path(subject, mode='wm')
    wm = nib.load(filepath_wm)
#     print(wm.affine)
    
    filepath_csf = get_file_path(subject, mode='csf')
    csf = nib.load(filepath_csf)
    
    func0 = func[0].slicer[:, :, :, 0]
    gm_func_size=nibp.resample_from_to(gm, func0)
    wm_func_size = nibp.resample_from_to(wm, func0)
    csf_func_size=nibp.resample_from_to(csf, func0)


    gm_values = gm_func_size.get_fdata()
    gm_mask = (gm_values>0.1)
    
    wm_values = wm_func_size.get_fdata()
    csf_values = csf_func_size.get_fdata()
    
    confounds_values = (wm_values+csf_values)/2
    confounds_mask = (confounds_values>0.1)
    
    subject_dir_name = 'sub-{:02d}_complete'.format(subject)
    save_dir = os.path.join(datapath, subject_dir_name, 'func', 'compcorr')
    print(save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for run in range(1, 9):
        print('subj: ', subject, 'run: ', run)
        at_func = deepcopy(func[run-1])
        
        func_values = at_func.get_fdata()
        func_reshaped = np.reshape(func_values,[at_func.shape[0]*at_func.shape[1]*at_func.shape[2],at_func.shape[3]])
        
        gm_reshaped = np.reshape(gm_mask,-1)
        confounds_mask_reshaped = np.reshape(confounds_mask,-1)
        
        func_gm = func_reshaped[gm_reshaped,:] # these are the functional data in gray matter
        func_confounds = func_reshaped[confounds_mask_reshaped,:] # these are the functional data in the regions of no interest

        func_gm = np.transpose(func_gm)
        func_confounds = np.transpose(func_confounds)
        # Fit PCA and extract PC timecourses
        pca = PCA(n_components = 5)
        pca.fit(func_confounds)
        confounds_pc = pca.fit_transform(func_confounds)
        
        linear = linear_model.LinearRegression()
        linear.fit(confounds_pc, func_gm)

        # predict the activity of each voxel for this run 
        predict = linear.predict(confounds_pc)
        func_denoised = func_gm - predict # t x v
        func_denoised = np.transpose(func_denoised) # v x t

        func_final = np.zeros([at_func.shape[0]*at_func.shape[1]*at_func.shape[2],at_func.shape[3]])
        func_final[gm_reshaped,:] = func_denoised
        func_final = np.reshape(func_final,[at_func.shape[0],at_func.shape[1],at_func.shape[2],at_func.shape[3]])

        new_header = at_func.header.copy()
        denoised_img = nib.nifti1.Nifti1Image(func_final, None, header=new_header)
        nib.save(denoised_img, os.path.join(save_dir,'func_run-{:02d}.nii.gz'.format(run)))



