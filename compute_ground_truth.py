import pickle
import numpy as np

def compute_average_ground_truth():
    """
    Computes the average ground truth precision across all cameras in the dataset.
    """
    print("Loading ground_truth_histograms.pkl...")
    with open("ground_truth_histograms.pkl", 'rb') as f:
        data = pickle.load(f)

    if not data:
        print("Error: ground_truth_histograms.pkl is empty.")
        return

    precisions = []
    for camera_info in data:
        tp_count = camera_info.get('tp_count', 0)
        fp_count = camera_info.get('fp_count', 0)
        total_count = tp_count + fp_count
        if total_count > 0:
            precision = tp_count / total_count
            precisions.append(precision)

    if not precisions:
        print("Error: No valid camera data found to compute average precision.")
        return

    average_precision = np.mean(precisions)
    
    print("\n✅ Ground Truth Calculation Complete:")
    print(f"   Recomputed Average Ground Truth Precision: {average_precision:.8f}")
    print(f"   Number of cameras used in calculation: {len(precisions)}")

if __name__ == "__main__":
    compute_average_ground_truth() 