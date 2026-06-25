import argparse
import os
from os.path import join as pjoin

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.glm.second_level import non_parametric_inference

MODULES = (
    'acceleration',
    'velocity',
    'control velocity predict acceleration',
    'control acceleration predict velocity',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Second-level permutation inference on group MNI t-maps.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'examples:\n'
            '  python SecondLevelPermutation.py --module acceleration\n'
            '  python SecondLevelPermutation.py --module velocity\n'
            '  python SecondLevelPermutation.py '
            '--module "control velocity predict acceleration"'
        ),
    )
    parser.add_argument(
        '--module',
        required=True,
        choices=MODULES,
        help='Encoding model module to run group inference on',
    )
    args = parser.parse_args(argv)
    return args.module

def main():
    # z-values
    subjects = [1,2,3,4,5,9,10,14,15,16,17,18,19,20]
    module = parse_args()

    map_name = 'wt_map'
    root_dir = '../results'
    corr_map_list = []
    for subject in subjects:
        subject_dir = pjoin(root_dir, 
                            'EncodingModel', 
                            'subject-{0:02d}'.format(subject), 
                            module)
        mni_tmap = nb.load(os.path.join(subject_dir, f'{map_name}.nii'))
        corr_map_list.append(mni_tmap)

    design_matrix = pd.DataFrame([1] * len(corr_map_list), columns=["intercept"])

    cluster_threshold = 0.0002 
    smoothing = 6
    n_perm = 10000
    out_dict = non_parametric_inference(
        second_level_input=corr_map_list,
        design_matrix=design_matrix,
        n_perm=n_perm,                       
        two_sided_test=False,
        smoothing_fwhm=smoothing,
        n_jobs=-1,
        threshold=cluster_threshold,
        verbose=1,
    )
    print((out_dict['logp_max_size'].get_fdata() >= -np.log10(.05)).sum()) 
    print((out_dict['logp_max_mass'].get_fdata() >= -np.log10(.05)).sum())
    print((out_dict['logp_max_t'].get_fdata() >= -np.log10(.05)).sum())


    # mni p-value map
    res_dir = os.path.join('../results/EncodingModel/Mean/MNI', module)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    res_path = os.path.join(res_dir, 
                            f'logp_max_size {map_name} {cluster_threshold} {smoothing}.nii')
    anat_path = '../forrest_study_fmri/sub-{0:02d}_complete/anat/sub-{1:02d}_T1w_preproc.nii'.format(1, 1)
    anat = nb.load(anat_path)
    new_header = anat.header.copy()
    nii = nb.nifti1.Nifti1Image(out_dict['logp_max_size'].get_fdata(), None, header=new_header) #logp_max_size
    nb.save(nii, res_path)

    # mni sig mask
    sig_mask = np.zeros_like(out_dict['logp_max_size'].get_fdata())
    sig_mask[out_dict['logp_max_size'].get_fdata() >= -np.log10(.05)] = 1

    res_path = os.path.join(res_dir, 
                            f'logp_max_size {n_perm} {map_name} {cluster_threshold} {smoothing} sig mask.nii')

    if not os.path.exists(res_path):
        nii = nb.nifti1.Nifti1Image(sig_mask, None, header=new_header) #logp_max_size
        nb.save(nii, res_path)


if __name__ == '__main__':
    main()