# Data Description

> Fill in the parts marked with `___` before submitting.

## Source
- Images of 8 microorganisms: ___ (own photographs / open-source dataset name and link)
- Collected / prepared by: ___

## Classes (8)

| # | Class name (as used in the file names) | Number of images |
|---|----------------------------------------|-----------------:|
| 1 | ___ | ___ |
| 2 | ___ | ___ |
| 3 | ___ | ___ |
| 4 | ___ | ___ |
| 5 | ___ | ___ |
| 6 | ___ | ___ |
| 7 | ___ | ___ |
| 8 | ___ | ___ |
| | **Total** | ___ |

(The Dataset tab of the program shows these counts, and the Export Report button writes them to `evaluation_report.txt`.)

## Format
- 100 x 100 pixels, grayscale, `.jpg` / `.jpeg` / `.png`
- File name pattern: `<microorganism name> (<number>).<extension>`, e.g. `paramecium (12).png`
- The class label is read from the part of the file name before the bracket.

## Preprocessing used by the program
1. Convert to grayscale and resize to the chosen input size (default 32 x 32 = 1,024 inputs; 100 x 100 = 10,000 inputs is also possible).
2. MinMax-scale every pixel to [0, 1] by dividing by 255.
3. Stratified train/test split (default 80 % / 20 %, per class, fixed seed).
4. Unreadable or corrupt files are skipped and reported.

## Target encoding
One output neuron per class; the correct class has target 1, all others 0 (one-hot).
