"""Common functions useful for multiple applications.
"""

import numpy as np
from load_trk_numba import load_streamlines
from scipy.io import loadmat
from benchmark_bundle_segmentation_dataset import get_tractogram_pathname as get_tractogram_pathname_bbs
from benchmark_bundle_segmentation_dataset import get_wmc_pathname as get_wmc_pathname_bbs
from benchmark_minor_bundle_segmentation_dataset import get_tractogram_pathname as get_tractogram_pathname_bmbs
from benchmark_minor_bundle_segmentation_dataset import get_wmc_pathname as get_wmc_pathname_bmbs


tractogram_cache = {}


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


if __name__=='__main__':

    from benchmark_bundle_segmentation_dataset import bundle_strings, dataset
    from time import time

    subject_id = 599469
    bundle_strings = ['IFO_left', 'AF_right']
    verbose = True

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
