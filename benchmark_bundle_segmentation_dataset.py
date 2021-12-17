"""Common data and functions to deal with the Benchmar
BundleSegmentation Dataset.

"""

import pathlib
from glob import glob


dataset = 'Benchmark_Bundle_Segmentation'
datadir = 'proj-605b72b0edebf063f4d274c4'


# These are the 105 subject IDs of the HCP dataset.
subject_ids = [599469, 599671, 601127, 613538, 620434, 622236, 623844,
               627549, 638049, 644044, 645551, 654754, 665254, 672756, 673455,
               677968, 679568, 680957, 683256, 685058, 687163, 690152, 695768,
               702133, 704238, 705341, 709551, 715041, 715647, 729254, 729557,
               732243, 734045, 742549, 748258, 748662, 749361, 751348, 753251,
               756055, 759869, 761957, 765056, 770352, 771354, 779370, 782561,
               784565, 786569, 789373, 792564, 792766, 802844, 814649, 816653,
               826353, 826454, 833148, 833249, 837560, 837964, 845458, 849971,
               856766, 857263, 859671, 861456, 865363, 871762, 871964, 872158,
               872764, 877168, 877269, 887373, 889579, 894673, 896778, 896879,
               898176, 899885, 901038, 901139, 901442, 904044, 907656, 910241,
               912447, 917255, 922854, 930449, 932554, 951457, 957974, 958976,
               959574, 965367, 965771, 978578, 979984, 983773, 984472, 987983,
               991267, 992774]


# These are the 72 bundle strings
bundle_strings = ['AF_left', 'CST_left', 'OR_left', 'ST_OCC_right',
                  'T_PAR_left', 'AF_right', 'CST_right', 'OR_right',
                  'ST_PAR_left', 'T_PAR_right', 'ATR_left', 'FPT_left',
                  'POPT_left', 'ST_PAR_right', 'T_POSTC_left',
                  'ATR_right', 'FPT_right', 'POPT_right',
                  'ST_POSTC_left', 'T_POSTC_right', 'CA', 'FX_left',
                  'SCP_left', 'ST_POSTC_right', 'T_PREC_left', 'CC_1',
                  'FX_right', 'SCP_right', 'ST_PREC_left',
                  'T_PREC_right', 'CC_2', 'ICP_left', 'SLF_III_left',
                  'ST_PREC_right', 'T_PREF_left', 'CC_3', 'ICP_right',
                  'SLF_III_right', 'ST_PREF_left', 'T_PREF_right',
                  'CC_4', 'IFO_left', 'SLF_II_left', 'ST_PREF_right',
                  'T_PREM_left', 'CC_5', 'IFO_right', 'SLF_II_right',
                  'ST_PREM_left', 'T_PREM_right', 'CC_6', 'ILF_left',
                  'SLF_I_left', 'ST_PREM_right', 'UF_left', 'CC_7',
                  'ILF_right', 'SLF_I_right', 'STR_left', 'UF_right',
                  'CC', 'MCP', 'ST_FO_left', 'STR_right', 'CG_left',
                  'MLF_left', 'ST_FO_right', 'T_OCC_left', 'CG_right',
                  'MLF_right', 'ST_OCC_left', 'T_OCC_right']


# These are the bundles whose tractogram was generated without ACT
noACT_list = ['CA', 'IFO_left', 'IFO_right', 'UF_left', 'UF_right']

def get_tractogram_pathname(subject_id, bundle_string):
    """Generate a valid pathname of a tractogram given subject_id and
    bundle_string of interest (to resolve ACT vs noACT).
    """
    global datadir
    ACT_string = 'ACT'
    if bundle_string in noACT_list:
        ACT_string = 'noACT'

    try:
        pathname = next(pathlib.Path(f'{datadir}/sub-{subject_id}/').glob(f'dt-neuro-track-trk.tag-{ACT_string}.id-*/track.trk'))
        return pathname
    except StopIteration:
        print('Tractogram not available!')
        raise FileNotFoundError


def get_wmc_pathname(subject_id, bundle_string):
    """Generate a valid pathname of a WMC file given subject_id and
    bundle_string (to resolve ACT vs noACT).

    The WMC file contrains the bundle-labels for each streamline of the
    corresponding tractogram.
    """
    global datadir
    ACT_string = 'ACT'
    if bundle_string in noACT_list:
        ACT_string = 'noACT'

    try:
        pathname = next(pathlib.Path(f'{datadir}/sub-{subject_id}/').glob(f'dt-neuro-wmc.tag-{ACT_string}.id-*/classification.mat'))
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
00: No bundle
01: AF_left        (Arcuate fascicle)
02: AF_right
03: ATR_left       (Anterior Thalamic Radiation)
04: ATR_right
05: CA             (Commissure Anterior)
06: CC_1           (Rostrum)
07: CC_2           (Genu)
08: CC_3           (Rostral body (Premotor))
09: CC_4           (Anterior midbody (Primary Motor))
10: CC_5           (Posterior midbody (Primary Somatosensory))
11: CC_6           (Isthmus)
12: CC_7           (Splenium)
13: CG_left        (Cingulum left)
14: CG_right
15: CST_left       (Corticospinal tract)
16: CST_right
17: MLF_left       (Middle longitudinal fascicle)
18: MLF_right
19: FPT_left       (Fronto-pontine tract)
20: FPT_right
21: FX_left        (Fornix)
22: FX_right
23: ICP_left       (Inferior cerebellar peduncle)
24: ICP_right
25: IFO_left       (Inferior occipito-frontal fascicle)
26: IFO_right
27: ILF_left       (Inferior longitudinal fascicle)
28: ILF_right
29: MCP            (Middle cerebellar peduncle)
30: OR_left        (Optic radiation)
31: OR_right
32: POPT_left      (Parieto‐occipital pontine)
33: POPT_right
34: SCP_left       (Superior cerebellar peduncle)
35: SCP_right
36: SLF_I_left     (Superior longitudinal fascicle I)
37: SLF_I_right
38: SLF_II_left    (Superior longitudinal fascicle II)
39: SLF_II_right
40: SLF_III_left   (Superior longitudinal fascicle III)
41: SLF_III_right
42: STR_left       (Superior Thalamic Radiation)
43: STR_right
44: UF_left        (Uncinate fascicle)
45: UF_right
46: CC             (Corpus Callosum - all)
47: T_PREF_left    (Thalamo-prefrontal)
48: T_PREF_right
49: T_PREM_left    (Thalamo-premotor)
50: T_PREM_right
51: T_PREC_left    (Thalamo-precentral)
52: T_PREC_right
53: T_POSTC_left   (Thalamo-postcentral)
54: T_POSTC_right
55: T_PAR_left     (Thalamo-parietal)
56: T_PAR_right
57: T_OCC_left     (Thalamo-occipital)
58: T_OCC_right
59: ST_FO_left     (Striato-fronto-orbital)
60: ST_FO_right
61: ST_PREF_left   (Striato-prefrontal)
62: ST_PREF_right
63: ST_PREM_left   (Striato-premotor)
64: ST_PREM_right
65: ST_PREC_left   (Striato-precentral)
66: ST_PREC_right
67: ST_POSTC_left  (Striato-postcentral)
68: ST_POSTC_right
69: ST_PAR_left    (Striato-parietal)
70: ST_PAR_right
71: ST_OCC_left    (Striato-occipital)
72: ST_OCC_right
"""
