# Dataset Description - Fruit Classification

This dataset consists of image samples across 5 distinct fruit classes used to train and evaluate the custom Multi-Layer Perceptron (MLP) neural network.

### Dataset Source and Preparation

The dataset used in this project was personally compiled by collecting fruit images through Google Images searches. The collected images were organized into five categories: Apple, Avocado, Banana, Pear, and Pineapple. To ensure consistency in the input dimensions and simplify image processing, all images were resized to 100 × 100 pixels and converted into grayscale format. These processed images were then used as the input data for the neural network during training and testing. This approach allowed the model to work with standardized image dimensions and intensity values while reducing the complexity of image representation.

---

## 1. Class Distribution

The dataset contains a total of **75 image samples** distributed across the following classes:

| Class Label | Image Count | Train Count (60%) | Test Count (40%) | Characteristics / Visual Features |
| :--- | :--- | :--- | :--- | :--- |
| **Apple** | 13 | 8 | 5 | Round contour, central stem indent, uniform shading |
| **Avocado** | 12 | 7 | 5 | Oval/pear shape, dark textured skin, distinct weight distribution |
| **Banana** | 21 | 13 | 8 | Elongated, curved aspect ratio, high contrast against backgrounds |
| **Pear** | 13 | 8 | 5 | Tapered neck, wide base, asymmetrical vertical contour |
| **Pineapple** | 16 | 10 | 6 | Diamond/grid texture, crown leaves, large rectangular aspect area |
| **Total** | **75** | **46** | **29** | *Stratified split preserves class proportions in train & test sets* |

---

## 2. Image Preprocessing & Feature Extraction

To prepare raw image files for input into the neural network engine, the following automated preprocessing pipeline is applied (`ImageDataset.cs`):

1. **Grayscale Conversion:** Full-color RGB images are converted into single-channel luminance values ($Y = 0.299R + 0.587G + 0.114B$).
2. **Resizing:** Images are scaled using bicubic interpolation to a standardized resolution of $100 \times 100$ pixels (total of $10,000$ input features/neurons).
3. **Normalization:** Pixel values ($0 \text{ to } 255$) are scaled using MinMax normalization to the continuous floating-point range $[0.0, 1.0]$ via $x_{norm} = \frac{\text{pixel}}{255.0}$.
4. **Label Encoding:** Targets are converted into one-hot encoded vectors of length 5 corresponding to the class index.

---

## 3. Data Splitting & Reproducibility

* **Stratified Split:** A fixed ratio (default 40% test split) is applied per class to guarantee balanced representation in both training and test evaluations.
* **Random Seed:** A fixed random seed (default `42`) ensures that train/test partition splits remain completely reproducible across training runs and hyperparameter grid searches.
