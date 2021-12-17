import pandas as pd


if __name__=='__main__':

    output_txt = True
    output_tex = True
    output_csv = True

    results_filenames = ['Benchmark_Bundle_Segmentation/results_streamlines_1NN_train_10.csv',
                         'Benchmark_Minor_Bundle_Segmentation/results_streamlines_1NN_train_10.csv']

    for results_filename in results_filenames:
        results = pd.read_csv(results_filename)
        # print(results)
        summary = results.groupby(['bundle_string']).agg({'DSC_voxels':['mean', 'std']})
        pd.set_option('display.max_rows', None)
        print(summary.sort_values(ascending=False, by=('DSC_voxels', 'mean')))

        if output_txt:
            report_filename = results_filename.replace(".csv", ".txt").replace("results", "report")
            print(f"Saving report in {report_filename}")
            f = open(report_filename, 'w')
            print(summary.sort_values(ascending=False, by=('DSC_voxels', 'mean')), file=f)
            f.close()

        if output_tex:
            report_filename = results_filename.replace(".csv", ".tex").replace("results", "report")
            print(f"Saving report in {report_filename}")
            f = open(report_filename, 'w')
            print(summary.sort_values(ascending=False, by=('DSC_voxels', 'mean')).round(2).to_latex(), file=f)
            f.close()

        if output_csv:
            report_filename = results_filename.replace("results", "report")
            print(f"Saving report in {report_filename}")
            f = open(report_filename, 'w')
            print(summary.sort_values(ascending=False, by=('DSC_voxels', 'mean')).to_csv(), file=f)
            f.close()

        print("")
