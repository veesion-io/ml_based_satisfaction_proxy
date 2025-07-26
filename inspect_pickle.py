import pickle
import numpy as np

with open("ground_truth_histograms.pkl", 'rb') as f:
    data = pickle.load(f)

if data:
    first_camera = data[0]
    print("Keys:", first_camera.keys())
    tp_density = np.array(first_camera['tp_density'])
    fp_density = np.array(first_camera['fp_density'])

    print("\nTP Density:", tp_density)
    print("TP Density Sum:", tp_density.sum())

    print("\nFP Density:", fp_density)
    print("FP Density Sum:", fp_density.sum())

    tp_count = tp_density.sum()
    fp_count = fp_density.sum()
    total_count = tp_count + fp_count
    gt_precision = tp_count / total_count if total_count > 0 else 0.0
    print(f"\nCalculated GT Precision: {gt_precision}")
else:
    print("Pickle file is empty.") 