# FruitClassifier (C# Multi-Layer Perceptron)

A Windows Forms application that classifies 100x100 grayscale images of **5 fruits** 
(*apple, avocado, banana, pear, pineapple*) using a 3-layer neural network (input -> hidden -> output) trained by back-propagation.
The network engine is built on the core structure (`BackNeural 1.cs`, `InputN 1.cs`, `HiddenN 1.cs`, `OuputN 1.cs`); learning rates are dynamically configurable.

---

## 1. How to Run

Requirements: Windows, Visual Studio 2019/2022 with the ".NET desktop development" workload (.NET Framework 4.8).

1. Open `BackPropNNImages.csproj` in Visual Studio.
2. Press **F5** (or Build > Start). No external NuGet packages are required.
3. The program creates an `outputs` folder next to the `.exe` (`bin\Debug\outputs`) for logs, reports, and weight files.

---

## 2. Image / File Naming

Recommended file name pattern:

```
<fruit name> (<number>).jpg | .jpeg | .png
```

Examples: `apple (1).png`, `banana (12).jpeg`. The text preceding the parenthesis is parsed as the class label. Sub-folders are also supported:

| File Name Example | Parsed Class Label |
| :--- | :--- |
| `apple(3).jpg`, `apple [3].png` | apple |
| `banana_3.jpg`, `banana-3.jpg`, `banana 3.jpg` | banana |
| `avocado (3) - Copy.jpg`, `avocado (3) (1).jpg` | avocado |
| `pineapple/12.png` | pineapple (folder name used) |

Images are loaded in grayscale and resized to the dimensions specified in the application (default 100x100).

---

## 3. Application Workflow (Step by Step)

| Step | Section | Description |
| :--- | :--- | :--- |
| 1 | *1 Load images* | Click **Open Folder...** (or **Select Images...**). Choose the test split ratio (default 40%) and target image dimensions. |
| 2 | *2 Create network* | Configure hidden neuron count, initial weight multiplier, and seed, then press **Create Network**. |
| 3 | *3 Train* | Set epoch count and press **Train**. Real-time loss and accuracy curves update per epoch. |
| 4 | *4 Evaluate & Save* | Click **Evaluate Test Set**, **Export Report...**, or **Save Weights...** / **Load Weights...**. |
| - | *Identify tab* | Load any single fruit image to inspect the activation levels across all 5 output neurons. |
| - | *Experiments tab* | Run automated hyperparameter grid searches across configurations. |

---

## 4. Output Files (Folder `outputs`)

| File Name | Content |
| :--- | :--- |
| `training_log_<time>.csv` | Per-epoch metrics: training loss, training accuracy, test loss, test accuracy, execution time. |
| `hyperparameter_experiments_<time>.csv` | Grid search results covering evaluated hyperparameter combinations. |
| `evaluation_report.txt` | Detailed breakdown: dataset distribution, model parameters, accuracy, MSE loss, confusion matrix, precision/recall/F1. |
| `*.weights` + `*.weights.meta` | Network weight binary file and metadata storing dimensions, neuron counts, and class labels. |

---

## 5. Architecture & Preprocessing

* **Network Topology:** `Backprop.NeuralNet(inputs, hidden, outputs)`. Configured with 10,000 inputs ($100 \times 100$ pixels), 16–64 hidden neurons, and 5 output neurons representing *apple, avocado, banana, pear, and pineapple*. Target vectors are encoded as one-hot arrays.
* **Loss Function:** Mean Squared Error (MSE) computed between output activations and target one-hot vectors.
* **Preprocessing (`ImageDataset.cs`):** 
  * Grayscale conversion and bicubic resizing to $100 \times 100$.
  * MinMax scaling normalized to range $[0, 1]$ (`value / 255.0`).
  * Stratified split (60% training, 40% test split preserving class proportions across all 5 categories).

---

## 6. Hyperparameter Tuning

### Tuning Results (Extracted from Experiments Grid Search)

| Hidden | Weight Multiplier | LR (Out / In) | Epochs | Train Acc. | Test Acc. | Test Loss | Key Observation |
| -----: | ----------------: | ------------: | -----: | ---------: | --------: | --------: | :--- |
| **16** | **0.5** | **0.05 / 0.031** | **40** | **76.09%** | **72.41%** | **0.1120** | Moderate capacity, baseline convergence |
| **32** | **0.5** | **0.05 / 0.031** | **40** | **95.65%** | **75.86%** | **0.0796** | Good balance, steady loss decline |
| **32** | **1.0** | **0.05 / 0.031** | **40** | **97.83%** | **79.31%** | **0.0841** | High test accuracy, slight overfitting |
| **64** | **0.5** | **0.05 / 0.031** | **40** | **93.48%** | **79.31%** | **0.0780** | **Top performing configuration** |
| **64** | **2.0** | **0.05 / 0.031** | **40** | **91.30%** | **75.86%** | **0.0860** | Larger initial weights slowed fine convergence |

**Chosen Configuration & Justification:**  
Configurations with **64 hidden neurons**, a weight multiplier of **0.5–1.0**, and conservative learning rates (**0.05 output / 0.031 input**) achieved the optimal trade-off between convergence rate and generalizability. Setting learning rates too high caused variance in test loss, whereas smaller initial weight bounds (Xavier/Glorot scaling) prevented hidden neuron saturation across $10,000$ input features. Extending training to ~100 epochs with 16 hidden neurons yielded **97.83% training accuracy** and **79.31% test accuracy**.

### Final Model Configuration and Selection

The initial hyperparameter experiments explored different hidden neuron counts and learning rate configurations. The 64-hidden-neuron configuration achieved the best results among the configurations tested during the initial grid search. However, the final evaluated model used 16 hidden neurons, an input learning rate of 0.031, and an output learning rate of 0.05. This configuration was selected as a more practical and manageable model for the relatively small dataset. The final model was trained for 104 epochs and achieved 97.83% training accuracy and 79.31% testing accuracy. The distinction between the initial experimental results and the final evaluated model reflects the configuration ultimately selected for the project's final implementation.

---

## 7. Final Model Results

* **Final Training Accuracy:** **97.83%** (Training MSE: `0.015989`)
* **Final Test Accuracy:** **79.31%** (Test MSE: `0.071437`)
* **Macro Precision / Recall / F1:** `0.800 / 0.760 / 0.768`

### Per-Class Evaluation Summary

| Fruit Class | Test Support | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Apple** | 5 | 1.000 | 0.600 | 0.750 |
| **Avocado** | 5 | 0.600 | 0.600 | 0.600 |
| **Banana** | 8 | 0.800 | 1.000 | 0.889 |
| **Pear** | 5 | 0.600 | 0.600 | 0.600 |
| **Pineapple** | 6 | 1.000 | 1.000 | 1.000 |

### Observations & Feature Analysis
1. **Distinctive Features:** **Pineapple** and **Banana** achieved perfect or near-perfect recall (100%), as their distinct aspect ratios and structural features convert cleanly into grayscale pixel patterns.
2. **Class Confusion:** Minor misclassifications occurred primarily between rounder shapes (**Apple**, **Avocado**, and **Pear**). Because pixel values are converted to single-channel grayscale, color cues (e.g., green pear vs. green apple) are lost, forcing the network to rely entirely on outer visual contours and shading patterns.

---

## 8. Source Files

| File | Purpose |
| :--- | :--- |
| `BackNeural 1.cs` | Engine class (learning rates made dynamic) |
| `InputN 1.cs`, `HiddenN 1.cs`, `OuputN 1.cs` | Neuron classes (**unchanged**) |
| `ImageDataset.cs` | Image parsing, resizing, normalisation, stratified splitting |
| `ClassifierModel.cs` | Model wrapper for initialization, training loops, evaluation, and weight persistence |
| `ClassificationMetrics.cs` | Metric computations (Accuracy, MSE, Confusion matrix, F1) |
| `HyperparameterExperiment.cs` | Automated grid search execution |
| `OutputFiles.cs` | Log and report exporters |
| `LineChart.cs`, `ProbabilityBars.cs` | Custom UI visualizer components |
| `Form1.cs`, `Form1.UI.cs`, `Form1.Designer.cs` | Application interface and interaction logic |
| `DATA_DESCRIPTION.md` | Dataset source and breakdown documentation |
