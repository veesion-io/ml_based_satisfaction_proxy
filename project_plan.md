# Project Plan: Density Prediction Model

This document outlines the corrected plan for building a model that predicts the probability densities of True Positives (TP) and False Positives (FP) for a camera based on a small sample of its alert probabilities.

### 1. The Ground Truth (Target Densities)

The ground truth target will be two **normalized density vectors**, pre-calculated for each camera using **all of its available alert data**.

*   **TP Density Vector:**
    1.  Collect all `max_proba` values for alerts that are True Positives.
    2.  Create a histogram of these probabilities with a fixed number of bins (e.g., 20) covering the range [0, 1].
    3.  **Normalize the histogram counts so the sum of all bin heights equals 1.** This resulting vector is the ground truth TP probability density.

*   **FP Density Vector:**
    1.  Repeat the exact same process for all alerts that are False Positives.
    2.  The result is the ground truth FP probability density.

These two normalized vectors will be the targets the model tries to predict.

### 2. The Model Input

The model's input will be a small, random sample of `max_proba` values from a single camera (e.g., 50 probabilities). The model will not be told which of these correspond to TPs or FPs.

### 3. The Model's Prediction (Output Densities)

The model will be a `DeepSets` architecture that processes the set of input probabilities and outputs two vectors of raw numbers (logits), one for the TP density and one for the FP density.

*   **Softmax Activation:** A **Softmax function** will be applied independently to each of the two output vectors.
*   **Result:** This ensures that each of the model's predictions is a valid probability distribution (the values in each vector sum to 1), perfectly matching the format of our ground truth density vectors.

### 4. The Loss Function

To measure how well the model's predicted distribution approximates the true ground truth distribution, the **Kullback-Leibler (KL) Divergence loss** will be used. This is the standard and most appropriate loss function for comparing two probability distributions.

### 5. The Evaluation Plots

The evaluation plots will be generated for each epoch and will provide a clear and direct visualization of the model's performance.

*   **X-axis:** Probability, ranging from 0 to 1.
*   **Y-axis:** Probability Density.
*   **Content:** For both TP and FP, the plot will show the ground truth density (e.g., as a blue bar chart) and the model's predicted density (e.g., as an overlapping red bar chart). This allows for an immediate visual comparison of the predicted distribution against the true distribution.

---

# Appendix: Original Project Plan (V1)

## 1. Objective

The primary goal of this initial phase is to process raw theft alert data from a source parquet file. We will transform this data by creating a unified binary label for theft events and structure the output for efficient, per-camera analysis in subsequent modeling stages.

## 2. Environment Setup

To ensure reproducibility, all work should be conducted within a dedicated virtual environment.

*   **Environment Name:** `satisfaction_proxy_env`
*   **Recommended Tool:** `conda` or `venv`

*   **To Create (using venv):**
    ```bash
    python3 -m venv satisfaction_proxy_env
    ```

*   **To Activate:**
    ```bash
    source satisfaction_proxy_env/bin/activate
    ```

*   **Required Python Libraries:**
    *   `pandas`: For data manipulation.
    *   `pyarrow` / `fastparquet`: For reading and writing parquet files.

    Install them using:
    ```bash
    pip install pandas pyarrow fastparquet
    ```

## 3. Step 1: Data Preprocessing and Structuring

### 3.1. Define Theft Labels

The following labels are considered indicators of theft. This dictionary will be used to map raw string labels to a binary outcome.

```python
LABELS_BSR_RATIOS = {
    "backpack": 0.0073773310368468945,
    "burst shot": 0.0023898990360455025,
    "consumption": 0.005768299027567761,
    "deblistering": 0.0038569216328267135,
    "gesture into body": 0.007369356383484813,
    "other suspicious": 0.002601605299705335,
    "personal bag": 0.008340375083555097,
    "product into stroller": 0.008280012187690432,
    "suspicious bag": 0.0033948719358686615,
    "theft": 0.050290877524495584,
    "suspicious": 0.003580911249770547,
}
THEFT_LABELS = list(LABELS_BSR_RATIOS.keys())
```

### 3.2. Processing Logic

1.  **Load Data:** Read the source parquet file containing the alert data into a pandas DataFrame.
2.  **Create Binary Label:**
    *   Add a new column to the DataFrame, named `is_theft`.
    *   For each row, check if the value in the `label` column is present in the `THEFT_LABELS` list.
    *   If the label is in the list, set `is_theft` to `1`.
    *   If the label is not in the list, set `is_theft` to `0`.
3.  **Select Relevant Data:** Create a new DataFrame containing only the essential columns for the next stage:
    *   `camera_id` (or equivalent columns like `store` and `camera`)
    *   `probability`
    *   `is_theft`
4.  **Save Processed Data:** Save the resulting DataFrame to a new parquet file. This single file will contain the data for all cameras, allowing for easy loading and filtering.

## 4. Step 2: Model Training Pipeline

### 4.1. Objective

To train a machine learning model capable of predicting the distribution of True Positive (TP) and False Positive (FP) alert counts from a given set of alerts from a single camera. The model must learn to generalize this prediction across unseen cameras.

### 4.2. Model Architecture

The model will consist of three main components, designed to handle variable-sized sets of inputs and predict a probability distribution.

1.  **DeepSets Encoder:**
    *   **Purpose:** To create a fixed-size representation of a variable-sized set of alert probabilities, ensuring the output is not affected by the order of alerts (permutation invariance).
    *   **Structure:** It will consist of two small MLPs:
        1.  A function `φ` (phi) that processes each alert probability individually.
        2.  An aggregation function (e.g., `sum` or `mean`) that combines the outputs of `φ`.

2.  **MLP Head:**
    *   **Purpose:** To take the fixed-size vector from the DeepSets encoder and transform it into a set of parameters for the output distribution.
    *   **Structure:** A standard multi-layer perceptron (`ρ`, rho).

3.  **Output Distribution Layer:**
    *   **Purpose:** To model the 2D probability distribution of (TP, FP) counts.
    *   **Architecture Options:**
        *   **Normalizing Flow:** A more flexible and powerful option. The MLP head would output the parameters for a base distribution (e.g., a simple Gaussian) and the sequence of transformations that define the flow. This can model complex, multi-modal distributions.
        *   **Gaussian Mixture Model (GMM):** A simpler, yet effective alternative. The MLP head would output the means, covariances, and mixture weights for a pre-defined number of 2D Gaussian components.

### 4.3. Data Generation and Training

1.  **Dataset Split:** Before training, split the full list of cameras into `train`, `validation`, and `test` sets. This is critical to ensure the model's evaluation reflects its ability to generalize to new, unseen cameras.

2.  **Training Sample Generation:**
    *   The training process will be based on generating batches of data on the fly.
    *   For each batch:
        1.  Select a camera from the `train` set.
        2.  Select a random subset size `k`.
        3.  Randomly sample `k` alerts from the selected camera's data.
        4.  The **input** to the model is the set of `k` alert probabilities.
        5.  The **ground truth** is the actual count of `(TPs, FPs)` for that sample, where a TP is an alert with `is_theft = 1` and an FP is an alert with `is_theft = 0`.

3.  **Loss Function:**
    *   The model will be trained by minimizing the **Negative Log-Likelihood (NLL)**.
    *   This loss measures how unlikely the model's predicted distribution considers the ground truth `(TP, FP)` count to be. The goal is to train the model to assign a high probability to the true outcome.

### 4.4. Evaluation

1.  **Evaluation Process:**
    *   The evaluation will be performed on the `test` set of cameras.
    *   For each camera in the test set:
        1.  Define a range of subset sizes to evaluate (e.g., `k = 10, 20, ..., N`, where N is the total number of alerts for that camera).
        2.  For each size `k`, repeatedly sample many subsets of alerts.
        3.  For each subset, get the **predicted distribution** from the model and the **ground truth (TP, FP) count**.

2.  **Metric: R² of Predicted vs. Ground Truth Curves**
    *   **Ground Truth Curve:** For each subset size `k`, calculate the average ground truth TP and FP counts over all the samples. This creates two curves: `E[TP_gt]` vs. `k` and `E[FP_gt]` vs. `k`.
    *   **Predicted Curve:** For each subset size `k`, calculate the expected TP and FP counts from the model's predicted distribution for each sample, and then average these expectations. This creates two predicted curves: `E[TP_pred]` vs. `k` and `E[FP_pred]` vs. `k`.
    *   The final metric is the **R² score** calculated between the ground truth curve and the predicted curve for both TPs and FPs across all evaluated subset sizes. This measures how well the model's average predictions match the true average outcomes at different scales. 