"""In this experiment, we load some example bundles except the one of
the target_subject. We pool them together as a single bundle and then
we do 1NN with respect to the target tractogram.
"""


import numpy as np
from tools import load_target, load_example_bundle, streamlines2mask
from time import time
import pandas as pd
from copy import copy
import pickle
from random import sample, seed
from nearest_neighbor import nearest_neighbor
from os import mkdir
from os.path import isdir


def segment_bundle(dataset, target_subject_id, bundle_string, subject_ids, nb_points=32, voxel_step=1.0/10.0, apply_affine=True, verbose=True):
    target_tractogram, target_bundle_indices_true, target_bundle_true, target_affine, target_volume_size = load_target(dataset, target_subject_id, bundle_string, nb_points, apply_affine)
    target_tractogram = target_tractogram.astype(np.float32)
    if len(target_bundle_indices_true) == 0:
        print(f"WARNING: {bundle_string} of taget subject {target_subject_id} is empty!")
        return None, None, None, None, None
        
    print("")
    example_bundle_all = []
    if num_examples == 'all':
        example_subject_ids = copy(subject_ids)
        example_subject_ids.remove(target_subject_id)
    elif type(num_examples) == int:
        example_subject_ids = copy(subject_ids)
        example_subject_ids.remove(target_subject_id)  # This ensures that training and test sets are disjoint
        example_subject_ids = sample(example_subject_ids, num_examples)
    else:
        raise Exception

    print(f"Loading examples of {bundle_string} for the training set:")
    for counter, example_subject_id in enumerate(example_subject_ids):
        print(f"{counter}) ID: {example_subject_id}")
        example_bundle, example_affine, example_volume_size = load_example_bundle(dataset, example_subject_id, bundle_string, nb_points, apply_affine, verbose=False)
        if len(example_bundle) == 0:
            print("WARNING: This bundle is empty, so skipping.")
            continue
        example_bundle = example_bundle.astype(np.float32)
        example_bundle_all.append(example_bundle)
        counter += 1

    example_bundle_all = np.vstack(example_bundle_all)
    print(f"example_bundle_all: {len(example_bundle_all)} stereamlines")

    print("")
    print("Computing the predicted bundle with 1-NN")
    t0 = time()
    distances, target_bundle_indices_predicted = nearest_neighbor(example_bundle_all, target_tractogram, verbose=verbose) # target_subject_id, bundle_string, verbose=verbose)
    print(f"{time()-t0} sec.")
    print(f"target_bundle_indices_predicted: {len(target_bundle_indices_predicted)} streamlines")

    target_bundle_indices_predicted_unique, idx, counts = np.unique(target_bundle_indices_predicted, return_index=True, return_counts=True)
    distances_unique = distances[idx]
    print(f"target_bundle_indices_predicted_unique: {len(target_bundle_indices_predicted_unique)} streamlines")
    
    print("Scores: dice similarity coefficient")
    dsc_streamlines = 2 * np.intersect1d(target_bundle_indices_true, target_bundle_indices_predicted_unique).size / (target_bundle_indices_true.size + target_bundle_indices_predicted_unique.size)
    print(f"DSC_streamlines = {dsc_streamlines}")

    target_bundle_predicted = target_tractogram[target_bundle_indices_predicted_unique]
    target_bundle_predicted_voxel_mask = streamlines2mask(target_bundle_predicted, target_affine, target_volume_size)
    target_bundle_true_voxel_mask = streamlines2mask(target_bundle_true, target_affine, target_volume_size)
    dsc_voxels = 2 * (target_bundle_predicted_voxel_mask * target_bundle_true_voxel_mask).sum() / (target_bundle_predicted_voxel_mask.sum() + target_bundle_true_voxel_mask.sum())
    print(f"DSC_voxels = {dsc_voxels}")

    return target_bundle_indices_predicted_unique, distances_unique, counts, dsc_streamlines, dsc_voxels


if __name__=='__main__':
    seed(0)

    from benchmark_bundle_segmentation_dataset import dataset, get_available_subjects, bundle_strings
    # from benchmark_minor_bundle_segmentation_dataset import dataset, get_available_subjects, bundle_strings

    print(f"Experiments segmenting bundles from the {dataset} dataset.")

    print(f"Creating directory {dataset} if not existing, to store results")
    if not isdir(dataset): mkdir(dataset)

    available_subject_ids = get_available_subjects()
    print(f"Available subjects IDs: {available_subject_ids}")

    target_subject_ids = available_subject_ids
    example_subject_ids = available_subject_ids

    num_examples = 10
    print(f"Using {num_examples} example bundles in the training set")

    # Technical parameters:
    nb_points = 16  # number of points for resampling
    verbose = True  # verbosity
    voxel_step = 1/10.0  # step size used to convert streamlines to voxels. Lower is better but more expensive.
    apply_affine = True  # Use affine available in tractograms.

    predicted_dir = f'{dataset}/predicted_bundles_train_{num_examples}/'
    if not isdir(predicted_dir): mkdir(predicted_dir)
    print(f"Saving all predicted bundles in {predicted_dir}")

    results_filename = f'{dataset}/results_streamlines_1NN_train_{num_examples}.csv'
    print(f"Saving results (scores) in {results_filename}")

    try:
        results = pd.read_csv(results_filename)
        print("Resuming results:")
        print(results)
    except:
        print('Starting a new table of results.')
        results = pd.DataFrame(columns=['target_subject_id', 'num_examples', 'bundle_string', 'DSC_streamlines', 'DSC_voxels'])
    

    print("")
    for target_subject_id in target_subject_ids:
        for bundle_string in bundle_strings:
            print(f"Segmenting {bundle_string}")
            # check if result is alaready available and, in that case, skip computation:
            row = results[(results['target_subject_id']==target_subject_id) & (results['num_examples']==num_examples) & (results['bundle_string']==bundle_string)]
            if len(row) > 0:
                print(row)
                continue

            # Perform Segmentation and get results:
            target_bundle_indices_predicted_unique, distances_unique, counts, dsc_streamlines, dsc_voxels = segment_bundle(dataset, target_subject_id, bundle_string, example_subject_ids, nb_points, voxel_step, apply_affine, verbose)

            # Check whether the target bundle was available. If not, skip last part of the code:
            if target_bundle_indices_predicted_unique is None:
                print(f"WARNING: The {bundle_string} of target subject {target_subject_id} is empty!")
                continue

            # Save the computed segmentation:
            target_bundle_indices_predicted_filename = 'target_%s_examples_%s_%s_1NN.pickle' % (target_subject_id, num_examples, bundle_string)
            print('Saving %s' % target_bundle_indices_predicted_filename)
            pickle.dump({"target_bundle_indices_predicted_unique": target_bundle_indices_predicted_unique,
                         "distances_unique": distances_unique,
                         "counts": counts},
                        open(predicted_dir + target_bundle_indices_predicted_filename, 'wb'))

            # Add results in the table of results:
            results = results.append({'target_subject_id': target_subject_id, 'num_examples': num_examples, 'bundle_string':bundle_string, 'DSC_streamlines':dsc_streamlines, 'DSC_voxels':dsc_voxels}, ignore_index=True)
            results.to_csv(results_filename, index=False)
            
            print("")
