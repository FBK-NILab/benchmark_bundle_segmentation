"""Common data and functions to deal with the minor bundles dataset.

"""

import pathlib
from glob import glob


dataset = 'Benchmark_Minor_Bundle_Segmentation'
datadir = 'proj-6058cd229feaae29c59d8aee'


subject_ids = [500222, 506234, 510326, 512835, 513130, 517239, 521331,
               522434, 523032, 524135, 525541, 529953, 531536, 531940,
               540436, 541640, 547046, 548250, 552241, 553344, 555348,
               555651, 555954, 557857, 558960, 559053, 561242, 561444,
               562345, 562446, 565452, 567052, 567759, 568963, 569965,
               573249, 573451, 578057, 580044, 580347, 580751, 581450,
               583858, 585256, 585862, 586460, 594156, 597869, 598568,
               800941, 803240, 809252, 810843, 814548, 814649, 815247,
               816653, 818455, 825048, 825553, 825654, 826353, 826454,
               827052, 832651, 833148, 833249, 835657, 837560, 837964,
               844961, 845458, 849264, 849971, 856766, 861456, 865363,
               869472, 872764, 873968, 877269, 884064, 885975, 886674,
               887373, 889579, 891667, 896778, 896879, 898176, 899885,
               901038, 901139, 901442, 905147, 907656, 910241, 910443,
               911849, 912447, 917255, 917558, 919966, 927359, 929464,
               947668, 951457, 957974, 965367, 970764, 972566, 973770,
               978578, 979984, 983773, 987074, 990366, 991267, 992673,
               995174, 996782]


bundle_strings= ['Left_pArc', 'Right_pArc',
                 'Left_TPC', 'Right_TPC',
                 'Left_MdLF-SPL', 'Right_MdLF-SPL',
                 'Left_MdLF-Ang', 'Right_MdLF-Ang']



def get_tractogram_pathname(subject_id):
    """Generate a valid pathname of a tractogram given subject_id and
    bundle_string of interest (to resolve ACT vs noACT).
    """
    global datadir
    try:
        pathname = next(pathlib.Path(f'{datadir}/sub-{subject_id}/').glob(f'dt-neuro-track-trk.tag-ensemble.tag-t1.id-*/track.trk'))
        return pathname
    except StopIteration:
        print(f'Tractogram not available!')
        raise FileNotFoundError


def get_wmc_pathname(subject_id, bundle_string):
    """Generate a valid pathname of a WMC file given subject_id and
    bundle_string (to resolve ACT vs noACT).

    The WMC file contrains the bundle-labels for each streamline of the
    corresponding tractogram.
    """
    global datadir
    try:
        pathname = next(pathlib.Path(f'{datadir}/sub-{subject_id}/').glob(f'dt-neuro-wmc.tag-ensemble.id-*/classification.mat'))
        return pathname
    except StopIteration:
        print('WMC file not available!')
        raise FileNotFoundError


def get_peaks_pathname(subject_id):
    """Generate a valid pathname of the peaks.nii.gz file given
    subject_id.
    """
    global datadir
    try:
        pathname = next(pathlib.Path(f'{datadir}/sub-{subject_id}/').glob(f'dt-neuro-peaks.id-*/peaks.nii.gz'))
        return pathname
    except StopIteration:
        print('peaks file not available!')
        raise FileNotFoundError


def get_available_subjects():
    """Get the list of subjects actually having tractogram, WMC, and peak
    files.
    """
    global datadir
    global subject_ids
    available_subject_ids = [int(pathname.split('-')[-1]) for pathname in glob(f'{datadir}/sub-??????')]
    available_subject_ids = set(available_subject_ids).intersection(set(subject_ids))
    return sorted(list(available_subject_ids))


"""
- Left and right posterior arcuate fasciculus (Left_pArc and
  Right_pArc).

- Left and right temporo-parietal connection to the superior parietal lobule
(Left_TPC and Right_TPC)

- Left and right middle longitudinal fasciculus-superior parietal
lobule component (Left_MdLF-SPL and Right_MdLF-SPL)

- Left and right middle longitudinal fasciculus-superior angular
gyrus component (Left_ MdLF-Ang and Right_MdLF- Ang)

"""

