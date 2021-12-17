import numpy as np
from common import load_tractogram, load_bundle, get_bundle_idxs


from load_trk_numba import load_streamlines
# from wasserthal_2018_common import subject_ids, bundle_strings, tractogram_pathname, bundle_pathname, bundle_indices_pathname
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.vox2track import streamline_mapping
from dipy.tracking.utils import subsegment, length
from scipy.spatial import KDTree
import pickle
import pandas as pd
from time import time



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


def load_target(dataset, target_subject_id, bundle_string, nb_points=32, apply_affine=True, verbose=True):
    """This function loads the desired tractogram and bundle, resample to
    the desired number of point and apply the affine if requested.
    """
    target_tractogram, header, lengths, idxs = load_tractogram(dataset, target_subject_id, bundle_string, verbose=verbose, container='list')
    print(f"{len(target_tractogram)} streamlines")
    print(f"Resampling all streamlines to {nb_points} points.")
    target_tractogram = np.array(set_number_of_points(target_tractogram, nb_points=nb_points))
    target_affine = header['voxel_to_rasmm']
    target_volume_size = header['dimensions']

    print("Loading indices of the true target bundle")
    target_bundle_indices_true = get_bundle_idxs(dataset, target_subject_id, bundle_string, verbose=verbose)
    target_bundle_true = target_tractogram[target_bundle_indices_true]
    print(f"{len(target_bundle_indices_true)} streamlines")
    return target_tractogram, target_bundle_indices_true, target_bundle_true, target_affine, target_volume_size
    
    
def load_example_bundle(dataset, example_subject_id, bundle_string, nb_points=32, apply_affine=True, verbose=True):
    """This function loads the desired bundle, resample to the desired
    number of point and apply the affine if requested.
    """
    example_bundle, header, lengths, idxs = load_bundle(dataset,
                                                        example_subject_id,
                                                        bundle_string,
                                                        apply_affine=apply_affine,
                                                        container='list',
                                                        verbose=verbose)
    example_affine = header['voxel_to_rasmm']
    example_volume_size = header['dimensions']
    print(f"{len(example_bundle)} streamlines")
    print(f"Resampling all streamlines to {nb_points} points.")
    example_bundle = np.array(set_number_of_points(example_bundle, nb_points=nb_points))
    return example_bundle, example_affine, example_volume_size
    

if __name__=='__main__':


    nb_points = 32
    voxel_step = 1/10.0
    verbose = True
    results_filename = 'results_1NN.csv'
    
    target_subject_id = subject_ids[0]
    example_subject_id = subject_ids[1]
    bundle_string = "CST_left" # bundle_strings[0]

    try:
        results = pd.read_csv(results_filename)
        print("Resuming results:")
        print(results)
    except:
        print('Starting a new table of results.')
        results = pd.DataFrame(columns=['target_subject_id', 'example_subject_id', 'bundle_string', 'DSC_streamlines', 'DSC_voxels'])
    
    for target_subject_id in subject_ids:
        target_tractogram_pathname = tractogram_pathname(target_subject_id, bundle_string)
        target_bundle_indices_pathname = bundle_indices_pathname(target_subject_id, bundle_string)
        print("Loading target tractogram")
        target_tractogram, header, lengths, idxs = load_streamlines(target_tractogram_pathname,
                                                                    apply_affine=True,
                                                                    container='list',
                                                                    verbose=verbose)
        target_affine = header['voxel_to_rasmm']
        target_volume_size = header['dimensions']
        print(f"{len(target_tractogram)} streamlines")
        print(f"Resampling all streamlines to {nb_points} points.")
        target_tractogram = np.array(set_number_of_points(target_tractogram, nb_points=nb_points))

        print("Loading indices of the true target bundle")
        print(target_bundle_indices_pathname)
        target_bundle_indices_true = np.load(target_bundle_indices_pathname)
        target_bundle_true = target_tractogram[target_bundle_indices_true]
        print(f"{len(target_bundle_indices_true)} streamlines")

        for example_subject_id in subject_ids:
            row = results[(results['target_subject_id']==target_subject_id) & (results['example_subject_id']==example_subject_id) & (results['bundle_string']==bundle_string)]
            if len(row) > 0:
                print(row)
                continue
            
            example_bundle_pathname = bundle_pathname(example_subject_id, bundle_string)

            print("")
            print("Loading example bundle")
            example_bundle, header, lengths, idxs = load_streamlines(example_bundle_pathname,
                                                                     apply_affine=True,
                                                                     container='list',
                                                                     verbose=verbose)
            example_affine = header['voxel_to_rasmm']
            example_volume_size = header['dimensions']
            print("Checking that example and target are compatible.")  # todo: AC-PC registration is not great, we need MNI registration.
            assert((example_affine == target_affine).all())
            assert((example_volume_size == target_volume_size).all())
            print(f"{len(example_bundle)} streamlines")
            print(f"Resampling all streamlines to {nb_points} points.")
            example_bundle = np.array(set_number_of_points(example_bundle, nb_points=nb_points))

            print("")
            print("Computing the predicted bundle with 1-NN")
            t0 = time()
            distances, target_bundle_indices_predicted = nearest_neighbor(example_bundle, target_tractogram, verbose=verbose)
            print(f"{time()-t0} sec.")

            print("Scores: dice similarity coefficient")
            print("See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient")
            dsc_streamlines = 2 * np.intersect1d(target_bundle_indices_true, target_bundle_indices_predicted).size / (target_bundle_indices_true.size + target_bundle_indices_predicted.size)
            print(f"DSC_streamlines = {dsc_streamlines}")

            target_bundle_predicted = target_tractogram[target_bundle_indices_predicted]
            target_bundle_predicted_voxel_mask = streamlines2mask(target_bundle_predicted, target_affine, target_volume_size)
            target_bundle_true_voxel_mask = streamlines2mask(target_bundle_true, target_affine, target_volume_size)
            dsc_voxels = 2 * (target_bundle_predicted_voxel_mask * target_bundle_true_voxel_mask).sum() / (target_bundle_predicted_voxel_mask.sum() + target_bundle_true_voxel_mask.sum())
            print(f"DSC_voxels = {dsc_voxels}")

            target_bundle_predicted_filename = 'target_%s_example_%s_%s_1NN.pickle' % (target_subject_id, example_subject_id, bundle_string)
            print('Saving %s' % target_bundle_predicted_filename)
            pickle.dump(target_bundle_predicted, open('predicted_bundles/' + target_bundle_predicted_filename, 'wb'))

            results = results.append({'target_subject_id': target_subject_id, 'example_subject_id': example_subject_id, 'bundle_string':bundle_string, 'DSC_streamlines':dsc_streamlines, 'DSC_voxels':dsc_voxels}, ignore_index=True)
            results.to_csv(results_filename, index=False)
            
            print("")
