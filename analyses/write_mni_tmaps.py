import argparse
import os
import shutil
from os.path import join as pjoin

import nibabel as nb
import nibabel.processing as nibp
import numpy as np
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.spm import Normalize12

def write_nii(subject, np_array, layer, res_dir):
    filepath_func = '../forrest_study_fmri/sub-{0:02d}_complete/func/sub-{1:02d}_ses-movie_task-movie_run-{2:d}_bold_space-T1w_preproc.nii.gz'.format(subject, subject, 1)
    func = nb.load(filepath_func)
    filepath_gm = '../forrest_study_fmri/sub-{0:02d}_complete/anat/sub-{1:02d}_T1w_class-GM_probtissue.nii.gz'.format(subject, subject)
    gm = nb.load(filepath_gm)
    func0 = func.slicer[:, :, :, 0]
    gm_func_size=nibp.resample_from_to(gm, func0)
    new_header = gm_func_size.header.copy()
    nii = nb.nifti1.Nifti1Image(np_array, gm_func_size.affine, header=new_header)
    ##
    # nii_resampled = nibp.resample_from_to(nii, gm)
    # nii = nii_resampled
    ##
    nb.save(nii, pjoin(res_dir, '{0}.nii'.format(layer)))
    return pjoin(res_dir, '{0}.nii'.format(layer))

def normalize_and_save(subject, subject_nii_path, res_dir, spm12_path):
    anat_dir = '../forrest_study_fmri/sub-{0:02d}_complete/anat/'.format(subject)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    template = '../forrest_study_fmri/MNI/mni_icbm152_nlin_asym_09c/1mm_T1.nii.gz'
    
    T1 = pjoin(anat_dir, 'sub-{0:02d}_T1w_preproc.nii.gz'.format(subject))
    gunzip = Gunzip(in_file=T1)
    gzip_res = gunzip.run()
    T1_filename = os.path.basename(gzip_res.outputs.out_file)
    T1_nii = shutil.move(gzip_res.outputs.out_file, os.path.join(res_dir, T1_filename))
        
    normalize = Normalize12(jobtype='estwrite',
                            tpm=template,
                            apply_to_files=subject_nii_path,
                            image_to_align=T1_nii,
                            write_voxel_sizes=[1, 1, 1],
                            paths=[spm12_path]
                           )
                            
    res = normalize.run()


ALLOWED_MODULES = ('velocity', 'acceleration')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Write subject t-maps to NIfTI and normalize to MNI.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'example:\n'
            '  python writeMNI_tMaps.py 1 --module velocity '
            '--spm12-path /path/to/spm12'
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
        help='Encoding module whose t_map.npy to convert (velocity or acceleration)',
    )
    parser.add_argument(
        '--spm12-path',
        required=True,
        help='Path to SPM12 installation directory',
    )
    args = parser.parse_args(argv)
    return args.subject, args.module, args.spm12_path


def main():
    subject, module_name, spm12_path = parse_args()
    layer_names = ['t_map'] 
    module_dir = f'{module_name}/segments/GLMResult'

    root_dir = '../results'
    in_map_dir = pjoin(root_dir, 
                    'EncodingModel', 
                    'subject-{0:02d}'.format(subject),
                    module_dir)
    res_dir = pjoin(root_dir, 
                    'EncodingModel', 
                    'subject-{0:02d}'.format(subject),
                    module_name)

    for map_name in layer_names:
        pcorr_map = np.load(pjoin(in_map_dir, f'{map_name}.npy'))
        mean_map = np.nanmean(pcorr_map, axis=0)
        
        nii_path = write_nii(subject, mean_map, f'{map_name}', res_dir)
        normalize_and_save(subject, nii_path, res_dir, spm12_path)


if __name__ == '__main__':
    main()