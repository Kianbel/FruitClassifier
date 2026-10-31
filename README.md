# MicroID - Microorganism Classifier (C# Multi-Layer Perceptron)

A Windows Forms application that classifies 100x100 grayscale microscope images of **8 microorganisms**
with a 3-layer neural network (input -> hidden -> output) trained by back-propagation.
The network engine is the one from the previous activity (`BackNeural 1.cs`, `InputN 1.cs`,
`HiddenN 1.cs`, `OuputN 1.cs`); the only edit is that the two learning rates can now be set (section 6).

---

## 1. How to run

Requirements: Windows, Visual Studio 2019/2022 with the ".NET desktop development" workload (.NET Framework 4.8).

1. Open `BackPropNNImages.csproj` in Visual Studio.
2. Press **F5** (or Build > Start). No NuGet packages are needed.
3. The program creates an `outputs` folder next to the `.exe` (`bin\Debug\outputs`) for logs, reports and weights.

## 2. Image / file naming

Recommended file name:

```
<microorganism name> (<number>).jpg | .jpeg | .png
```

Examples: `paramecium (1).png`, `euglena (23).jpeg`. The text before the number is the class label, so all images of
one organism must use the same name. Sub-folders are allowed. These styles are also understood
(upper/lower case does not matter, `_` counts as a space):

| File name | Class |
|-----------|-------|
| `euglena(3).jpg`, `euglena [3].png` | euglena |
| `euglena_3.jpg`, `euglena-3.jpg`, `euglena 3.jpg`, `euglena3.jpg` | euglena |
| `euglena (3) - Copy.jpg`, `euglena (3) (1).jpg` | euglena |
| `Volvox_colony (12).png` | Volvox colony |
| `amoeba\12.png` (name is only a number) | amoeba (the folder name is used) |

A class name that itself ends in a number (e.g. `h1n1`) must be written with brackets: `h1n1 (4).png`.
Images are 100x100 grayscale; they are resized to the size chosen in the app (default 32x32).

## 3. Using the application (step by step)

| Step | Where | What to do |
|------|-------|-----------|
| 1 | *1 Load images* | **Open Folder...** (or **Select Images...**). Choose the test split (default 20 %) and the image size. |
| 2 | *2 Create the network* | Choose hidden neurons, initial weight multiplier and seed, press **Create Network**. |
| 3 | *3 Train* | Choose the number of epochs, press **Train**. Loss and accuracy curves update after every epoch. |
| 4 | *4 Evaluate and save* | **Evaluate Test Set**, **Export Report...**, **Save Weights...** / **Load Weights...** |
| - | *Identify* tab | Open any single image and see the activation of each of the 8 output neurons. |
| - | *Experiments* tab | Systematic hyper-parameter comparison (see section 6). |

## 4. Output files (folder `outputs`)

| File | Content |
|------|---------|
| `training_log_<time>.csv` | One row per epoch: train loss, train accuracy, test loss, test accuracy, seconds. |
| `hyperparameter_experiments_<time>.csv` | One row per tested configuration. |
| `evaluation_report.txt` | Dataset, settings, accuracy, MSE, confusion matrix, precision/recall/F1, training history. |
| `*.weights` + `*.weights.meta` | Saved network. The `.weights` file is written by the teacher's `saveWeights()`; the small `.meta` file stores image size, hidden neurons and class names. **Keep both files together.** |

## 5. Design

**Network.** `Backprop.NeuralNet(inputs, hidden, outputs)`: sigmoid neurons, online (per-image) back-propagation.
Inputs = image side x image side. Outputs = 8 (one per class); the target is a one-hot vector.
`ClassifierModel.cs` is the only class that talks to the engine.

**Loss.** Mean squared error between the 8 output activations and the one-hot target (averaged over images and outputs).
Training loss is the running loss while the epoch is processed; test loss is computed on the held-out images after every epoch.

**Preprocessing (`ImageDataset.cs`).**
- Class label parsed from the file name; corrupt/unreadable files are skipped and listed in the Dataset tab.
- Image -> grayscale -> resized (bicubic) -> **MinMax scaled to [0,1]** (`value / 255`).
- **Stratified** train/test split (every class keeps the same ratio, at least one test image per class when the class has 2+ images), fixed split seed so results are reproducible.

**Weight initialisation.** The teacher's classes start with weights in [-0.5, 0.5]. With 1,000+ inputs this saturates every
hidden neuron immediately. The initial weights are therefore generated in a Xavier/Glorot range
(`m * sqrt(6 / (fan_in + fan_out))`, `m` = *initial weight multiplier*) and handed to the engine through its own
`loadWeights()` method, so the engine files stay untouched.

**Evaluation (`ClassificationMetrics.cs`).** Accuracy, MSE, confusion matrix, per-class precision / recall / F1 and their macro averages.
A class that is never predicted gets precision 0 instead of dividing by zero.

## 6. Hyper-parameter tuning

Use the **Experiments** tab: enter lists such as hidden neurons `16, 32, 64, 128`, weight multipliers `0.5, 1, 2`,
learning rates `0.05, 0.1, 0.2`, epochs `10, 20, 40`. One network is trained per (hidden, multiplier, learning rate) combination up to the largest epoch value and evaluated
at every epoch checkpoint, always with the same seed and the same split. The table is saved as CSV and the best row
(highest test accuracy, ties -> lowest test loss) is highlighted; **Use Best Settings** copies it to the controls.

**Learning rates.** Step 2 has one learning rate for the hidden->output weights (default 0.2) and one for the
input->hidden weights (default 0.15). In the Experiments tab the *Learning rates* list is the output-layer rate; the
input-layer rate is always that value times the ratio set in step 2 (default 0.15 / 0.2 = 0.75).

> **Change made to the teacher's file (the only one).** In `BackNeural 1.cs` the two learning rates were `const`.
> They are now ordinary private fields (same names, same default values) and a small method
> `setLearningRates(double outputRate, double inputRate)` was added. Nothing else in that file was changed.

### Tuning results (fill in after running the Experiments tab)

| Hidden | Weight multiplier | LR out / in | Epochs | Train acc. | Test acc. | Test loss | Comment |
|-------:|------------------:|------------:|-------:|-----------:|----------:|----------:|---------|
|        |                   |             |        |            |           |           |         |

**Chosen configuration and why:** _(write 3-4 sentences: which settings won, what happened with too few / too many hidden
neurons, what the loss curves showed)_

## 7. Results (fill in)

- Final training accuracy: ___ %
- Final test accuracy: ___ %
- Observations (which organisms are confused with each other, over-fitting, etc.): ___

## 8. Source files

| File | Purpose |
|------|---------|
| `BackNeural 1.cs` | Teacher's network class. Only change: learning rates are adjustable (see section 6) |
| `InputN 1.cs`, `HiddenN 1.cs`, `OuputN 1.cs` | Teacher's neuron classes (**unchanged**) |
| `ImageDataset.cs` | File-name parsing, image loading, normalisation, train/test split |
| `ClassifierModel.cs` | Wrapper around the engine: init, train epoch, predict, evaluate, save/load weights |
| `ClassificationMetrics.cs` | Accuracy, MSE, confusion matrix, precision/recall/F1 |
| `HyperparameterExperiment.cs` | Grid search over hidden size, weight multiplier, epochs |
| `OutputFiles.cs` | CSV writers (training log, experiments) |
| `LineChart.cs`, `ProbabilityBars.cs` | Self-drawn charts (no extra libraries) |
| `Form1.cs`, `Form1.UI.cs`, `Form1.Designer.cs` | Program logic and user interface |
| `DATA_DESCRIPTION.md` | Description of the image data |
