import numpy as np
from common import load_peaks, get_bundle_mask
import nibabel as nib
from scipy.spatial import KDTree
from time import time
from copy import copy
from random import sample
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from random import seed
from os import mkdir
from os.path import isdir


if __name__=='__main__':
    seed(0)

    from benchmark_bundle_segmentation_dataset import dataset, get_available_subjects, bundle_strings
    # from benchmark_minor_bundle_segmentation_dataset import dataset, get_available_subjects, bundle_strings

    print(f"Experiments segmenting bundles from the {dataset} dataset using voxels.")

    print(f"Creating directory {dataset} if not existing, to store results")
    if not isdir(dataset): mkdir(dataset)

    available_subject_ids = get_available_subjects()
    print(f"Available subjects IDs: {available_subject_ids}")

    target_subject_ids = available_subject_ids
    example_subject_ids_all = available_subject_ids

    num_examples = 10
    print(f"Using {num_examples} example bundles in the training set")

    n_neighbors = 1
    print(f"Using KNeighborsClassifier with k={n_neighbors}")

    # Technical parameters:
    nb_points = 16  # number of points for resampling
    verbose = True  # verbosity
    voxel_step = 1/10.0  # step size used to convert streamlines to voxels. Lower is better but more expensive.
    apply_affine = True  # Use affine available in tractograms.
    alpha = 1.0  # coefficient to balance ijk values with peak values during 1NN

    predicted_dir = f'{dataset}/predicted_bundles_train_{num_examples}_voxels/'
    if not isdir(predicted_dir): mkdir(predicted_dir)
    print(f"Saving all predicted bundles in {predicted_dir}")

    results_filename = f'{dataset}/results_voxels_1NN_train_{num_examples}.csv'
    print(f"Saving results (scores) in {results_filename}")

    try:
        results = pd.read_csv(results_filename)
        print("Resuming results:")
        print(results)
    except:
        print('Starting a new table of results.')
        results = pd.DataFrame(columns=['target_subject_id', 'num_examples', 'bundle_string', 'DSC_voxels'])
    

    for target_subject_id in target_subject_ids:
        
        target_peaks, target_peaks_vol = load_peaks(dataset, target_subject_id, alpha=1.0)
        print("Preparing data for 1NN: transforming volumes to vectors")
        
        ijk = np.where((~np.isnan(target_peaks_vol)).any(axis=3))
        tmp = target_peaks_vol[ijk]
        tmp = np.nan_to_num(tmp, 0.0)  # set the remaining nan to 0.0 (Pietro says mrtrix put some nans for very low values)
        X_target =  np.hstack([np.array(ijk).T, tmp])
        print(f"X_target: {X_target.shape}")
        # print(f"X_target.max(0): {X_target.max(0)}")

        for bundle_string in bundle_strings:
            print(f"Segmenting {bundle_string}")
            # check if result is alaready available and, in that case, skip computation:
            row = results[(results['target_subject_id']==target_subject_id) & (results['num_examples']==num_examples) & (results['bundle_string']==bundle_string)]
            if len(row) > 0:
                print(row)
                continue

            print(f"Loading target bundle mask")
            target_bundle_mask_vol = get_bundle_mask(dataset, target_subject_id, bundle_string, voxel_step=voxel_step)

            example_subject_ids = copy(example_subject_ids_all)
            example_subject_ids.remove(target_subject_id)
            example_subject_ids = sample(example_subject_ids, num_examples)

            print(f"Loading {len(example_subject_ids)} examples:")
            X_example1 = []
            X_example0 = []
            size_example1 = []
            size_example0 = []
            for i, example_subject_id in enumerate(example_subject_ids):

                example_peaks, example_peaks_vol = load_peaks(dataset, example_subject_id, alpha=1.0)

                print(f"Loading example bundle mask")
                example_bundle_mask_vol = get_bundle_mask(dataset, example_subject_id, bundle_string, voxel_step=voxel_step)
                if example_bundle_mask_vol.sum() == 0.0:
                    print("WARNING: This bundle is empty!")
                    continue

                # size = int(example_bundle_mask_vol.sum())
                ijk1 = np.where(example_bundle_mask_vol[:,:,:] > 0.0)
                tmp1 = example_peaks_vol[ijk1]
                tmp1 = np.nan_to_num(tmp1, 0.0)  # set the remaining nan to 0.0 (Pietro says mrtrix put some nans for very low values)
                tmp1 = np.hstack([np.array(ijk1).T, tmp1])
                X_example1.append(tmp1)
                size1 = len(tmp1)
                print(f"1: {size1} voxels")
                size_example1.append(size1)

                ijk0 = np.where((~np.isnan(example_peaks_vol)).any(axis=3) * (~(example_bundle_mask_vol > 0.0)))  # this is the brain minus the bundle
                tmp0 = example_peaks_vol[ijk0]
                tmp0 = np.nan_to_num(tmp0, 0.0)  # set the remaining nan to 0.0 (Pietro says mrtrix put some nans for very low values)
                tmp0 = np.hstack([np.array(ijk0).T, tmp0])
                kdt0 = KDTree(tmp0)
                distance, index = kdt0.query(tmp1, k=50)
                index_unique = np.unique(np.concatenate(index))
                tmp0 = tmp0[index_unique]  # restrict tmp0 to the k-nn of the bundle (outside the bundle)
                X_example0.append(tmp0)
                size0 = len(tmp0)
                print(f"0: {size0} voxels")
                size_example0.append(size0)

            size_example1 = np.array(size_example1)
            size_example0 = np.array(size_example0)
            X_example = np.vstack(X_example1 + X_example0)
            print(f"X_example: {X_example.shape}")
            y_example = np.concatenate([np.ones(size_example1.sum()),
                                        np.zeros(size_example0.sum())])
            print(f"y_example: {y_example.shape}")

            clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='kd_tree', n_jobs=-1)
            print(clf)
            print("Fit")
            clf.fit(X_example, y_example)
            print("Predict")
            y_target = clf.predict(X_target)

            target_bundle_mask_predicted_vol = np.zeros(target_peaks_vol.shape[:3])
            tmp = X_target[y_target==1][:, :3].astype(int).T
            target_bundle_mask_predicted_vol[tmp[0], tmp[1], tmp[2]] = 1.0

            DSC_voxels = 2.0 * (target_bundle_mask_predicted_vol * target_bundle_mask_vol).sum() / (target_bundle_mask_predicted_vol.sum() + target_bundle_mask_vol.sum())
            print(f"DSC_voxels = {DSC_voxels}")
            print("")
            print("")

            results = results.append({'target_subject_id': target_subject_id, 'num_examples': num_examples, 'bundle_string':bundle_string, 'DSC_voxels':DSC_voxels}, ignore_index=True)
            results.to_csv(results_filename, index=False)
            
