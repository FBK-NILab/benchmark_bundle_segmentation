# benchmark_bundle_segmentation
Code for benchmarking automatic bundle segmentation methods.

1. Download data from [`brainlife.io`](https://brainlife.io). The two datasets addressed so far are the [Benchmark Bundle Segmentation Dataset](https://doi.org/10.25663/brainlife.pub.29) and the [Benchmark Minor Bundle Segmentation Dataset](https://doi.org/10.25663/brainlife.pub.28). You can download even just a portion of the datasets, i.e. just a few subjects, as long as you download all files associated to each subject (TRK, peaks, WMC). In the end, you need a directory (or link to) `proj-605b72b0edebf063f4d274c4` for the first dataset and `proj-6058cd229feaae29c59d8aee` for the second one.
2. Execute `experiment_voxels.py` or `experiment_streamlines.py` to run the two baselines methods. In the first lines of the `__main__` section, it is indicated on which of the two datasets the baselines are run. The required packages to be installed to run the experiments are listed in `requirements.txt`.
3. Execute `summary_of_results.py` in order to obtain a short summary of the performance of the two methods on the two datasets with mean and standard deviation of the Dice Similarity Coefficient (DSC) on the predicted vs. expert-curated voxel masks of white matter bunldes. The summary output is in three formats: `.txt`, `.csv`, `.tex`.

## Download data from `brainlife.io`
The [Benchmark Bundle Segmentation Dataset](https://doi.org/10.25663/brainlife.pub.29) and the [Benchmark Minor Bundle Segmentation Dataset](https://doi.org/10.25663/brainlife.pub.28) are published on `brainlife.io`. After clicking on the respective links, follow the Download button there and select all datatypes, i.e., track/trk, wmc, peaks, as well some or all subjects. Notice that, in order to download the data, you need to register or, alternatively, to obtain guest credentials. To fully reproduce the results of the benchmarks, the data of all subjects needs to be downloaded (740Gb). Nevertheless, the code in this repository runs also on a subset of all subjects, with which the two baseline methods will approximate the results of the benchmarks, to a good degree. In this last case, please condider to download at least 11 subjects per dataset. The download procedure on `brainlife.io` will produce a bash command, with a temporary unique token, to download all the desired files. Once the datasets - or portion of them - are downloaded, two new directories should be available: `proj-605b72b0edebf063f4d274c4` related to the [Benchmark Bundle Segmentation Dataset](https://doi.org/10.25663/brainlife.pub.29) and `proj-6058cd229feaae29c59d8aee` related to the [Benchmark Minor Bundle Segmentation Dataset](https://doi.org/10.25663/brainlife.pub.28)

## Execute the Experiments
The two baseline algorithms for automatic white matter bundle segmentation are implemented in `experiment_voxels.py`, which uses a voxel-based approach, and `experiment_streamlines.py`, which uses a streamline-based approach, both based on the idea of nearest neighbor. When executed, each file creates a directory for the results, if not already available, and a detailed log of the quality of segmentation for each bundle of each subject in the selected dataset. The choice of the dataset on which to run the segmentation is made with an import statement at the beginning of the `__main__` section of the code:
```
from benchmark_bundle_segmentation_dataset import dataset, get_available_subjects, bundle_strings
```
to use the [Benchmark Bundle Segmentation Dataset](https://doi.org/10.25663/brainlife.pub.29) and
```
from benchmark_bundle_segmentation_dataset import dataset, get_available_subjects, bundle_strings
```
to use the [Benchmark Minor Bundle Segmentation Dataset](https://doi.org/10.25663/brainlife.pub.28). Some parameters can be set in each experiment file, such as the number of subjects for the training set (`num_examples`), or the number of point for resampling streamlines (`nb_points`).
