"""Common functions useful for multiple applications.
"""

import numpy as np
from load_trk_numba import load_streamlines
from scipy.io import loadmat
from benchmark_bundle_segmentation_dataset import get_peaks_pathname as get_peaks_pathname_bbs
from benchmark_bundle_segmentation_dataset import get_tractogram_pathname as get_tractogram_pathname_bbs
from benchmark_bundle_segmentation_dataset import get_wmc_pathname as get_wmc_pathname_bbs
from benchmark_minor_bundle_segmentation_dataset import get_peaks_pathname as get_peaks_pathname_bmbs
from benchmark_minor_bundle_segmentation_dataset import get_tractogram_pathname as get_tractogram_pathname_bmbs
from benchmark_minor_bundle_segmentation_dataset import get_wmc_pathname as get_wmc_pathname_bmbs
import nibabel as nib
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.vox2track import streamline_mapping
from dipy.tracking.utils import subsegment, length
import zlib
from io import BytesIO


tractogram_cache = {}

bundle_mask_cache = {}


def get_peaks_pathname(dataset, subject_id):
    """Reroute the call to the related dataset-specific function.
    """
    if dataset == 'Benchmark_Bundle_Segmentation':
        return get_peaks_pathname_bbs(subject_id)
    elif dataset == 'Benchmark_Minor_Bundle_Segmentation':
        return get_peaks_pathname_bmbs(subject_id)
    else:
        print(dataset)
        raise NotImplementedError


def get_tractogram_pathname(dataset, subject_id, bundle_string=None):
    """Reroute the call to the related dataset-specific function.
    """
    if dataset == 'Benchmark_Bundle_Segmentation':
        return get_tractogram_pathname_bbs(subject_id, bundle_string)
    elif dataset == 'Benchmark_Minor_Bundle_Segmentation':
        return get_tractogram_pathname_bmbs(subject_id)
    else:
        print(dataset)
        raise NotImplementedError


def get_wmc_pathname(dataset, subject_id, bundle_string):
    """Reroute the call to the related dataset-specific function.
    """
    if dataset == 'Benchmark_Bundle_Segmentation':
        return get_wmc_pathname_bbs(subject_id, bundle_string)
    elif dataset == 'Benchmark_Minor_Bundle_Segmentation':
        return get_wmc_pathname_bmbs(subject_id, bundle_string)
    else:
        raise NotImplementedError


def get_labels_names(wmc_pathname, verbose=False):
    """From the given WMC pathname, extract bundle-labels for each
    streamline and the list of bundle_strings in the same order of the
    bundle-labels.

    Note: the WMC (White Matter Classification) file is a simple
    matlab file choose by Brainlife as a standard for
    streamline-related information.

    """
    if verbose:
        print(f"Loading {wmc_pathname}")

    data = loadmat(wmc_pathname)
    tmp = data['classification'][0][0]
    bundle_names = [bn[0] for bn in tmp[0][0]]
    if verbose:
        print(f"{len(bundle_names)} bundle_names")

    streamline_labels = tmp[1].squeeze()
    return streamline_labels, bundle_names


def get_bundle_idxs(dataset, subject_id, bundle_string, verbose=False):
    wmc_pathname = get_wmc_pathname(dataset, subject_id, bundle_string)
    streamline_labels, bundle_names = get_labels_names(wmc_pathname, verbose=verbose)
    bundle_label = bundle_names.index(bundle_string) + 1  # +1 because label '0' means 'no bundle'
    idxs = np.where(streamline_labels == bundle_label)[0]
    return idxs


def load_bundle(dataset, subject_id, bundle_string, apply_affine=True,
                container='array', verbose=False):
    """Load the desired bundle by extracting the specific streamlines from
    the related tractogram. The correct streamlines IDs are obtained
    from the related WMC file.
    """
    if verbose:
        print(f"Loading {bundle_string}")

    idxs = get_bundle_idxs(dataset, subject_id, bundle_string, verbose=verbose)
    if verbose:
        print(f"Loading {len(idxs)} streamlines")
    tractogram_pathname = get_tractogram_pathname(dataset, subject_id, bundle_string)
    streamlines, header, lengths, idxs = load_streamlines(tractogram_pathname,
                                                          idxs,
                                                          apply_affine=apply_affine,
                                                          container=container,
                                                          verbose=verbose)
    if verbose:
        print(f"Loaded {len(streamlines)} streamlines")

    return streamlines, header, lengths, idxs


def load_tractogram(dataset, subject_id, bundle_string,
                    apply_affine=True, container='array', verbose=False):
    """Load tractogram keeping the last one in cache.
    """
    global tractogram_cache
    tractogram_pathname = get_tractogram_pathname(dataset, subject_id, bundle_string)
    if tractogram_pathname in tractogram_cache.keys():
        print(f"{tractogram_pathname} in cache")
        print("Retrieving tractogram from cache!")
        streamlines, header, lengths, idxs = tractogram_cache[tractogram_pathname]
    else:
        streamlines, header, lengths, idxs = load_streamlines(tractogram_pathname,
                                                              idxs=None,
                                                              apply_affine=apply_affine,
                                                              container=container,
                                                              verbose=verbose)
        # We store just the last tractogram:
        tractogram_cache = {tractogram_pathname: (streamlines, header, lengths, idxs)}

    if verbose:
        print(f"Loaded {len(streamlines)} streamlines")

    return streamlines, header, lengths, idxs


def load_peaks(dataset, subject_id, alpha=1.0):
    peaks_filename = get_peaks_pathname(dataset, subject_id)
    print(f"Loading {peaks_filename}")
    peaks = nib.Nifti1Image.load(peaks_filename)
    peaks_vol = peaks.get_fdata() * alpha
    return peaks, peaks_vol


def streamlines2mask(streamlines, affine, volume_size, voxel_step=1/10.0):
    """Transform streamlines into the corresponding voxel mask.
    """
    if type(volume_size) is str and volume_size=='':
        print("WARNING: Missing volume size in header. Using default volume size from HCP data: (145, 174, 145)")
        volume_size = (145, 174, 145)

    mask = np.zeros(volume_size, dtype=np.float32)

    voxel_size = np.abs(affine.diagonal())[:3][0]
    max_segment_length = voxel_size * voxel_step
    # slower:
    # streamlines_denser = list(subsegment(streamlines, max_segment_length))
    # a bit faster:
    nb_points = (np.array(list(length(streamlines))) / max_segment_length).astype(int)
    streamlines_denser = [set_number_of_points(streamlines[i], nb_points[i]) for i in range(len(streamlines))]

    voxels_indices = list(streamline_mapping(streamlines_denser, affine).keys())
    for vi in voxels_indices:
        mask[vi] = 1.0

    return mask


def get_bundle_mask(dataset, subject_id, bundle_string, voxel_step=1/10):
    """Compute the bundle mask for the required bundle from the
    streamlines. Store the mask (compressed) in a cache for future
    use.

    Note: compression should help a lot in the case of the binary mask
    of a bundle, reaching 99% compression. Thousands of bundle masks
    can be cached in less than 100Mb of RAM. For this reason, the
    cache has no check on how many compress bundle masks it stores.

    """
    global bundle_mask_cache
    key = (dataset, subject_id, bundle_string, voxel_step)
    if key in bundle_mask_cache.keys():
        print(f"Retrieving mask from cache with key {key}")
        tmp = bundle_mask_cache[key]
        tmp.seek(0)  # This is necessary for the BytesIO container
        return np.load(tmp)['mask']
    else:
        streamlines, header, lengths, idxs = load_bundle(dataset, subject_id, bundle_string, apply_affine=True, verbose=True)
        affine = header['voxel_to_rasmm']
        volume_size = header['dimensions']
        mask = streamlines2mask(streamlines, affine, volume_size, voxel_step=voxel_step)
        print(f"Storing mask in cache with key {key}")
        tmp = BytesIO()
        np.savez_compressed(tmp, mask=mask)
        bundle_mask_cache[key] = tmp
        return mask


if __name__=='__main__':

    from benchmark_bundle_segmentation_dataset import bundle_strings, dataset
    from time import time

    subject_id = 599469
    bundle_strings = ['IFO_left', 'AF_right']
    verbose = True
    voxel_step = 1/10

    for i, bundle_string in enumerate(bundle_strings):
        print(i, bundle_string)
        t0 = time()
        streamlines, header, lengths, idxs = load_bundle(dataset, subject_id, bundle_string, verbose=verbose)
        print(f"Total time: {time() - t0} sec.")
        print("")

    t0 = time()
    streamlines, header, lengths, idxs = load_tractogram(dataset, subject_id, bundle_string, verbose=verbose)
    print(f"Total time: {time() - t0} sec.")

    print("")
    t0 = time()
    streamlines, header, lengths, idxs = load_tractogram(dataset, subject_id, bundle_string, verbose=verbose)
    print(f"Total time: {time() - t0} sec.")

    print("")
    t0 = time()
    peaks, peaks_vol = load_peaks(dataset, subject_id, alpha=1.0)
    print(f"Total time: {time() - t0} sec.")

    print("")
    t0 = time()
    mask = get_bundle_mask(dataset, subject_id, bundle_string, voxel_step=voxel_step)
    print(f"Total time: {time() - t0} sec.")

    print("")
    t0 = time()
    mask2 = get_bundle_mask(dataset, subject_id, bundle_string, voxel_step=voxel_step)
    print(f"Total time: {time() - t0} sec.")

    assert((mask == mask2).all())
