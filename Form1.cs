using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace BackPropNNImages
{
    /// <summary>
    /// Form1 (part 1): program logic.
    ///
    /// Workflow:  1) load images -> 2) create network -> 3) train -> 4) evaluate / save weights,
    /// then identify single images or run hyper-parameter experiments.
    /// The neural network itself is the teacher's Backprop.NeuralNet, used through ClassifierModel.
    /// </summary>
    public partial class Form1 : Form
    {
        /// <summary>Fixed seed so the train/test split is identical on every run.</summary>
        private const int SplitSeed = 2024;

        private ImageDataset dataset;
        private List<ImageSample> trainSet = new List<ImageSample>();
        private List<ImageSample> testSet = new List<ImageSample>();
        private ClassifierModel model;

        private readonly List<EpochResult> history = new List<EpochResult>();
        private ClassificationMetrics lastTrainMetrics;
        private ClassificationMetrics lastTestMetrics;
        private string lastSettingsText = "No network trained yet.";

        private readonly List<ExperimentResult> experimentResults = new List<ExperimentResult>();
        private ExperimentResult bestExperiment;

        private CancellationTokenSource activeCts;
        private bool isBusy;
        private bool closeRequested;
        private Control[] lockableControls;

        public Form1()
        {
            InitializeComponent();
            BuildUi();
            WireEvents();
            UpdateArchitectureLabel();
            ResetEvaluationViews();
        }

        private void WireEvents()
        {
            btnLoadFolder.Click += LoadFolder_Click;
            btnLoadFiles.Click += LoadFiles_Click;
            numTestPercent.ValueChanged += TestPercent_ValueChanged;
            btnCreate.Click += Create_Click;
            btnTrain.Click += Train_Click;
            btnStop.Click += Stop_Click;
            btnStopExperiments.Click += Stop_Click;
            btnOpenOutputs.Click += OpenOutputs_Click;
            btnEvaluate.Click += Evaluate_Click;
            btnExport.Click += Export_Click;
            btnSaveWeights.Click += SaveWeights_Click;
            btnLoadWeights.Click += LoadWeights_Click;
            btnOpenImage.Click += Identify_Click;
            btnRunExperiments.Click += RunExperiments_Click;
            btnApplyBest.Click += ApplyBest_Click;
            FormClosing += Form1_FormClosing;

            // Everything that must not be touched while a long operation is running.
            lockableControls = new Control[]
            {
                btnLoadFolder, btnLoadFiles, numTestPercent, cmbImageSize,
                numHidden, numInitMultiplier, numLrOutput, numLrInput, numSeed, btnCreate,
                numEpochs, numLogEvery, btnTrain,
                btnEvaluate, btnExport, btnSaveWeights, btnLoadWeights,
                btnOpenImage,
                txtExpHidden, txtExpMultipliers, txtExpLearningRates, txtExpEpochs, btnRunExperiments, btnApplyBest
            };
        }

        // =====================================================================
        // Step 1: load images
        // =====================================================================

        private async void LoadFolder_Click(object sender, EventArgs e)
        {
            using (FolderBrowserDialog dialog = new FolderBrowserDialog())
            {
                dialog.Description = "Choose the folder that contains the microorganism images (sub-folders are included).";
                if (dialog.ShowDialog(this) != DialogResult.OK) return;

                string[] files;
                try
                {
                    files = Directory.GetFiles(dialog.SelectedPath, "*.*", SearchOption.AllDirectories);
                }
                catch (Exception ex)
                {
                    ShowError("The folder could not be read", ex);
                    return;
                }
                await LoadImagesAsync(files);
            }
        }

        private async void LoadFiles_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Title = "Select the microorganism images";
                dialog.Multiselect = true;
                dialog.Filter = "Images (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png";
                if (dialog.ShowDialog(this) != DialogResult.OK) return;
                await LoadImagesAsync(dialog.FileNames);
            }
        }

        private async Task LoadImagesAsync(IEnumerable<string> files)
        {
            List<string> list = files.Where(ImageDataset.IsSupported).ToList();
            if (list.Count == 0)
            {
                MessageBox.Show(this, "No .jpg, .jpeg or .png images were found.", "MicroID",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            int side = SelectedImageSide();
            progressBar.Style = ProgressBarStyle.Blocks;
            progressBar.Minimum = 0;
            progressBar.Maximum = list.Count;
            progressBar.Value = 0;
            SetBusy(true, "Loading " + list.Count + " images...");

            try
            {
                Action<int, int> report = delegate (int done, int total)
                {
                    InvokeOnUi(delegate { progressBar.Value = Math.Min(done, progressBar.Maximum); });
                };
                ImageDataset loaded = await Task.Run(() => ImageDataset.Load(list, side, report));

                if (loaded.Samples.Count == 0)
                {
                    MessageBox.Show(this, "None of the images could be read.\n\n" +
                        string.Join("\n", loaded.SkippedFiles.Take(10)), "MicroID",
                        MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
                if (loaded.ClassNames.Count < 2)
                {
                    MessageBox.Show(this, "Only one class was found (\"" + loaded.ClassNames[0] + "\").\n\n" +
                        "File names must look like  name (1).png  so that the class can be read from them.",
                        "MicroID", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                dataset = loaded;
                ApplySplit();
                RefreshDatasetViews();

                string note = "";
                if (model != null && !model.IsCompatibleWith(dataset))
                {
                    model = null;
                    UpdateArchitectureLabel();
                    ResetTrainingViews();
                    ResetEvaluationViews();
                    note = " The previous network does not fit these images, so create a new one.";
                }

                tabs.SelectedTab = tabDataset;
                SetStatus("Loaded " + dataset.Samples.Count + " images in " + dataset.ClassNames.Count +
                          " classes (" + trainSet.Count + " training / " + testSet.Count + " test)." +
                          (note.Length > 0 ? note : " Step 2: create the network."));
            }
            catch (Exception ex)
            {
                ShowError("The images could not be loaded", ex);
            }
            finally
            {
                SetBusy(false, null);
            }
        }

        private void TestPercent_ValueChanged(object sender, EventArgs e)
        {
            if (dataset == null || isBusy) return;
            ApplySplit();
            RefreshDatasetViews();
            SetStatus("Train/test split changed. Train the network again to get fair test results.");
        }

        private void ApplySplit()
        {
            List<ImageSample> train;
            List<ImageSample> test;
            dataset.Split((double)numTestPercent.Value / 100.0, SplitSeed, out train, out test);
            trainSet = train;
            testSet = test;
        }

        private int SelectedImageSide()
        {
            int index = cmbImageSize.SelectedIndex;
            if (index < 0 || index >= ImageSizes.Length) index = Array.IndexOf(ImageSizes, 32);
            return ImageSizes[index];
        }

        private void RefreshDatasetViews()
        {
            int[] all = dataset.CountPerClass(dataset.Samples);
            int[] train = dataset.CountPerClass(trainSet);
            int[] test = dataset.CountPerClass(testSet);

            lblDatasetSummary.Text = dataset.Samples.Count + " images, " + dataset.ClassNames.Count + " classes.\n" +
                trainSet.Count + " for training, " + testSet.Count + " for testing.\n" +
                "Input size: " + dataset.ImageSide + " x " + dataset.ImageSide + " = " + dataset.InputCount.ToString("N0") + " pixels." +
                (dataset.SkippedFiles.Count > 0 ? "\n" + dataset.SkippedFiles.Count + " file(s) skipped (see Dataset tab)." : "");

            gridClasses.Rows.Clear();
            for (int c = 0; c < dataset.ClassNames.Count; c++)
                gridClasses.Rows.Add(dataset.ClassNames[c], all[c], train[c], test[c]);
            gridClasses.Rows.Add("Total", dataset.Samples.Count, trainSet.Count, testSet.Count);
            gridClasses.Rows[gridClasses.Rows.Count - 1].DefaultCellStyle.Font =
                new Font("Segoe UI Semibold", 9.5f);

            txtSkipped.Text = dataset.SkippedFiles.Count == 0
                ? "None - every image was read successfully."
                : string.Join(Environment.NewLine, dataset.SkippedFiles);

            ClearThumbnails();
            for (int c = 0; c < dataset.ClassNames.Count; c++)
            {
                ImageSample sample = dataset.Samples.FirstOrDefault(s => s.ClassIndex == c);
                if (sample == null) continue;

                Panel item = new Panel();
                item.Size = new Size(104, 122);
                item.Margin = new Padding(4);

                PictureBox picture = new PictureBox();
                picture.Size = new Size(96, 96);
                picture.Location = new Point(4, 2);
                picture.SizeMode = PictureBoxSizeMode.Zoom;
                picture.BackColor = Color.FromArgb(240, 243, 248);
                picture.Image = ImageDataset.PixelsToBitmap(sample.Pixels, dataset.ImageSide);

                Label name = new Label();
                name.Text = dataset.ClassNames[c];
                name.AutoEllipsis = true;
                name.TextAlign = ContentAlignment.MiddleCenter;
                name.Size = new Size(104, 22);
                name.Location = new Point(0, 98);
                name.ForeColor = TextDark;

                item.Controls.Add(picture);
                item.Controls.Add(name);
                flowThumbs.Controls.Add(item);
            }
        }

        private void ClearThumbnails()
        {
            foreach (Control item in flowThumbs.Controls)
            {
                foreach (Control child in item.Controls)
                {
                    PictureBox picture = child as PictureBox;
                    if (picture != null && picture.Image != null) picture.Image.Dispose();
                }
            }
            flowThumbs.Controls.Clear();
        }

        // =====================================================================
        // Step 2: create the network
        // =====================================================================

        private async void Create_Click(object sender, EventArgs e)
        {
            if (dataset == null)
            {
                Info("Load your images first (step 1).");
                return;
            }

            int hidden = (int)numHidden.Value;
            double multiplier = (double)numInitMultiplier.Value;
            double lrOutput = (double)numLrOutput.Value;
            double lrInput = (double)numLrInput.Value;
            int seed = (int)numSeed.Value;
            ImageDataset ds = dataset;

            SetIndeterminate(true);
            SetBusy(true, "Creating the network...");
            try
            {
                ClassifierModel created = await Task.Run(() =>
                    ClassifierModel.Create(ds.ImageSide, hidden, ds.ClassNames, multiplier, lrOutput, lrInput, seed));

                model = created;
                history.Clear();
                lastTrainMetrics = null;
                lastTestMetrics = null;
                lastSettingsText = "Network " + created.InputCount + " -> " + hidden + " -> " + created.OutputCount +
                    ", initial weight multiplier " + multiplier.ToString("0.##", CultureInfo.InvariantCulture) +
                    ", learning rates " + lrOutput.ToString("0.###", CultureInfo.InvariantCulture) + " (output) / " +
                    lrInput.ToString("0.###", CultureInfo.InvariantCulture) + " (input)" +
                    ", seed " + seed + ", not trained yet.";
                ResetTrainingViews();
                ResetEvaluationViews();
                UpdateArchitectureLabel();
                LogLine("Network created: " + created.InputCount + " inputs, " + hidden + " hidden neurons, " +
                        created.OutputCount + " outputs (initial weight multiplier " +
                        multiplier.ToString("0.##", CultureInfo.InvariantCulture) + ", learning rates " +
                        lrOutput.ToString("0.###", CultureInfo.InvariantCulture) + " / " +
                        lrInput.ToString("0.###", CultureInfo.InvariantCulture) + ", seed " + seed + ").");
                SetStatus("Network created. Step 3: press Train.");
            }
            catch (Exception ex)
            {
                ShowError("The network could not be created", ex);
            }
            finally
            {
                SetIndeterminate(false);
                SetBusy(false, null);
            }
        }

        private void UpdateArchitectureLabel()
        {
            if (model == null)
            {
                lblArchitecture.Text = "No network yet. Create one, or load saved weights.";
                return;
            }
            long weights = (long)model.InputCount * model.HiddenCount + (long)model.HiddenCount * model.OutputCount;
            lblArchitecture.Text = "Network: " + model.InputCount.ToString("N0") + " inputs -> " +
                model.HiddenCount + " hidden -> " + model.OutputCount + " outputs (" +
                weights.ToString("N0") + " weights).\nClasses: " + string.Join(", ", model.ClassNames);
        }

        // =====================================================================
        // Step 3: train
        // =====================================================================

        private async void Train_Click(object sender, EventArgs e)
        {
            if (!EnsureModelMatchesDataset()) return;
            if (trainSet.Count == 0)
            {
                Info("There are no training images.");
                return;
            }

            ClassifierModel net = model;
            List<ImageSample> train = trainSet;
            List<ImageSample> test = testSet;
            int epochs = (int)numEpochs.Value;
            int logEvery = (int)numLogEvery.Value;
            int seed = (int)numSeed.Value;

            history.Clear();
            ResetTrainingViews();
            tabs.SelectedTab = tabTraining;

            lastSettingsText = "Network " + net.InputCount + " -> " + net.HiddenCount + " -> " + net.OutputCount +
                ", image size " + net.ImageSide + "x" + net.ImageSide +
                ", learning rates " + net.LearningRateOutput.ToString("0.###", CultureInfo.InvariantCulture) + " (output) / " +
                net.LearningRateInput.ToString("0.###", CultureInfo.InvariantCulture) + " (input)" +
                ", requested epochs " + epochs +
                ", seed " + seed + ", training images " + train.Count + ", test images " + test.Count + ".";
            LogLine("Training started. " + lastSettingsText);
            LogLine("Loss = mean squared error between the 'one-hot' target and the output neurons.");

            CancellationTokenSource cts = new CancellationTokenSource();
            activeCts = cts;
            progressBar.Style = ProgressBarStyle.Blocks;
            progressBar.Minimum = 0;
            progressBar.Maximum = epochs;
            progressBar.Value = 0;
            SetBusy(true, "Training...");

            OutputFiles.TrainingLogWriter writer = null;
            try
            {
                string logPath = OutputFiles.NewFilePath("training_log", ".csv");
                writer = new OutputFiles.TrainingLogWriter(logPath);
                OutputFiles.TrainingLogWriter csv = writer;

                Action<EpochResult> onEpoch = delegate (EpochResult r)
                {
                    csv.Append(r);
                    InvokeOnUi(delegate { HandleEpoch(r, epochs, logEvery); });
                };

                List<EpochResult> finished = await Task.Run(() => net.Train(train, test, epochs, seed, onEpoch, cts.Token));
                bool stopped = cts.IsCancellationRequested;

                SetStatus("Computing the final results...");
                progressBar.Style = ProgressBarStyle.Marquee;
                ClassificationMetrics trainMetrics = null;
                ClassificationMetrics testMetrics = null;
                await Task.Run(delegate
                {
                    trainMetrics = net.Evaluate(train);
                    testMetrics = net.Evaluate(test);
                });
                progressBar.Style = ProgressBarStyle.Blocks;

                lastTrainMetrics = trainMetrics;
                lastTestMetrics = testMetrics;
                ShowEvaluation();

                LogLine(stopped
                    ? "Training stopped by the user after " + finished.Count + " epoch(s)."
                    : "Training finished after " + finished.Count + " epoch(s).");
                LogLine("Training log saved to: " + logPath);
                lastSettingsText += " Epochs trained: " + finished.Count + ".";

                string summary = BuildFinalSummary(trainMetrics, testMetrics);
                SetStatus((stopped ? "Training stopped. " : "Training complete. ") + summary);
                tabs.SelectedTab = tabEvaluation;
                if (!closeRequested)
                {
                    MessageBox.Show(this,
                        (stopped ? "Training stopped.\n\n" : "Training complete!\n\n") + summary +
                        "\n\nThe epoch-by-epoch log was saved to:\n" + logPath,
                        "MicroID", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            catch (Exception ex)
            {
                ShowError("Training failed", ex);
            }
            finally
            {
                if (writer != null) writer.Dispose();
                activeCts = null;
                cts.Dispose();
                progressBar.Style = ProgressBarStyle.Blocks;
                SetBusy(false, null);
            }
        }

        private void HandleEpoch(EpochResult r, int epochs, int logEvery)
        {
            history.Add(r);
            chartLoss.AddPoint(lossTrainSeries, r.TrainLoss);
            chartLoss.AddPoint(lossTestSeries, r.TestLoss);
            chartAccuracy.AddPoint(accTrainSeries, r.TrainAccuracy);
            chartAccuracy.AddPoint(accTestSeries, r.TestAccuracy);
            progressBar.Value = Math.Min(r.Epoch, progressBar.Maximum);

            SetStatus(string.Format(CultureInfo.InvariantCulture,
                "Training...  epoch {0} of {1}   train loss {2:0.0000}", r.Epoch, epochs, r.TrainLoss));

            if (r.Epoch == 1 || r.Epoch % logEvery == 0 || r.Epoch == epochs)
                LogLine(FormatEpoch(r));
        }

        private static string FormatEpoch(EpochResult r)
        {
            CultureInfo inv = CultureInfo.InvariantCulture;
            string testLoss = double.IsNaN(r.TestLoss) ? "   -    " : r.TestLoss.ToString("0.000000", inv);
            string testAcc = double.IsNaN(r.TestAccuracy) ? "  -  " : (r.TestAccuracy * 100.0).ToString("0.0", inv) + "%";
            return "epoch " + r.Epoch.ToString(inv).PadLeft(4) +
                "  train loss " + r.TrainLoss.ToString("0.000000", inv) +
                "  train acc " + ((r.TrainAccuracy * 100.0).ToString("0.0", inv) + "%").PadLeft(6) +
                "  test loss " + testLoss +
                "  test acc " + testAcc.PadLeft(6);
        }

        private string BuildFinalSummary(ClassificationMetrics trainMetrics, ClassificationMetrics testMetrics)
        {
            CultureInfo inv = CultureInfo.InvariantCulture;
            string text = "Training accuracy " + (trainMetrics.Accuracy * 100.0).ToString("0.0", inv) + "%";
            if (testMetrics != null)
                text += ", test accuracy " + (testMetrics.Accuracy * 100.0).ToString("0.0", inv) + "%";
            return text + ".";
        }

        private void Stop_Click(object sender, EventArgs e)
        {
            if (activeCts != null)
            {
                activeCts.Cancel();
                SetStatus("Stopping after the current step...");
            }
        }

        private void OpenOutputs_Click(object sender, EventArgs e)
        {
            try
            {
                Process.Start(OutputFiles.GetOutputFolder());
            }
            catch (Exception ex)
            {
                ShowError("The output folder could not be opened", ex);
            }
        }

        private void ResetTrainingViews()
        {
            chartLoss.ClearData();
            chartAccuracy.ClearData();
            txtLog.Clear();
            progressBar.Value = 0;
        }

        // =====================================================================
        // Step 4: evaluate / report / save / load
        // =====================================================================

        private async void Evaluate_Click(object sender, EventArgs e)
        {
            if (!EnsureModelMatchesDataset()) return;
            if (testSet.Count == 0)
            {
                Info("There are no test images. Increase the test split.");
                return;
            }

            ClassifierModel net = model;
            List<ImageSample> train = trainSet;
            List<ImageSample> test = testSet;

            SetIndeterminate(true);
            SetBusy(true, "Evaluating...");
            try
            {
                ClassificationMetrics trainMetrics = null;
                ClassificationMetrics testMetrics = null;
                await Task.Run(delegate
                {
                    trainMetrics = net.Evaluate(train);
                    testMetrics = net.Evaluate(test);
                });
                lastTrainMetrics = trainMetrics;
                lastTestMetrics = testMetrics;
                ShowEvaluation();
                tabs.SelectedTab = tabEvaluation;
                SetStatus("Evaluation finished. " + BuildFinalSummary(trainMetrics, testMetrics));
            }
            catch (Exception ex)
            {
                ShowError("The evaluation failed", ex);
            }
            finally
            {
                SetIndeterminate(false);
                SetBusy(false, null);
            }
        }

        private void ResetEvaluationViews()
        {
            lblEvalSummary.Text = "No evaluation yet. Train the network or press 'Evaluate Test Set'.";
            gridConfusion.Columns.Clear();
            gridConfusion.Rows.Clear();
            gridMetrics.Rows.Clear();
        }

        private void ShowEvaluation()
        {
            ClassificationMetrics shown = lastTestMetrics ?? lastTrainMetrics;
            if (shown == null || model == null)
            {
                ResetEvaluationViews();
                return;
            }

            CultureInfo inv = CultureInfo.InvariantCulture;
            string set = lastTestMetrics != null ? "Test" : "Training";
            string text = set + " accuracy " + (shown.Accuracy * 100.0).ToString("0.0", inv) + "%     " +
                          "MSE " + shown.Mse.ToString("0.0000", inv) + "     " +
                          "macro F1 " + shown.MacroF1.ToString("0.000", inv) + "     (" + shown.SampleCount + " images)";
            if (lastTestMetrics != null && lastTrainMetrics != null)
                text += "\nTraining accuracy " + (lastTrainMetrics.Accuracy * 100.0).ToString("0.0", inv) +
                        "%   training MSE " + lastTrainMetrics.Mse.ToString("0.0000", inv);
            lblEvalSummary.Text = text;

            // confusion matrix
            gridConfusion.Columns.Clear();
            gridConfusion.Rows.Clear();
            DataGridViewTextBoxColumn first = new DataGridViewTextBoxColumn();
            first.HeaderText = "actual \\ predicted";
            first.FillWeight = 150f;
            first.SortMode = DataGridViewColumnSortMode.NotSortable;
            gridConfusion.Columns.Add(first);
            foreach (string name in model.ClassNames)
            {
                DataGridViewTextBoxColumn column = new DataGridViewTextBoxColumn();
                column.HeaderText = name;
                column.ToolTipText = name;
                column.FillWeight = 100f;
                column.SortMode = DataGridViewColumnSortMode.NotSortable;
                column.DefaultCellStyle.Alignment = DataGridViewContentAlignment.MiddleCenter;
                gridConfusion.Columns.Add(column);
            }
            int n = model.ClassNames.Count;
            for (int r = 0; r < n; r++)
            {
                object[] cells = new object[n + 1];
                cells[0] = model.ClassNames[r];
                for (int c = 0; c < n; c++) cells[c + 1] = shown.Confusion[r, c];
                int rowIndex = gridConfusion.Rows.Add(cells);
                for (int c = 0; c < n; c++)
                {
                    DataGridViewCell cell = gridConfusion.Rows[rowIndex].Cells[c + 1];
                    if (shown.Confusion[r, c] == 0) cell.Style.ForeColor = Color.FromArgb(170, 178, 190);
                    else if (r == c) cell.Style.BackColor = Color.FromArgb(214, 243, 222);
                    else cell.Style.BackColor = Color.FromArgb(252, 224, 224);
                }
            }

            // per-class metrics
            gridMetrics.Rows.Clear();
            for (int c = 0; c < n; c++)
            {
                gridMetrics.Rows.Add(model.ClassNames[c], shown.Support[c],
                    shown.Precision[c].ToString("0.000", inv),
                    shown.Recall[c].ToString("0.000", inv),
                    shown.F1[c].ToString("0.000", inv));
            }
            int macroRow = gridMetrics.Rows.Add("Macro average", shown.SampleCount,
                shown.MacroPrecision.ToString("0.000", inv),
                shown.MacroRecall.ToString("0.000", inv),
                shown.MacroF1.ToString("0.000", inv));
            gridMetrics.Rows[macroRow].DefaultCellStyle.Font = new Font("Segoe UI Semibold", 9.5f);
        }

        private void Export_Click(object sender, EventArgs e)
        {
            if (model == null || (lastTestMetrics == null && lastTrainMetrics == null))
            {
                Info("There are no results to export yet. Train or evaluate the network first.");
                return;
            }

            using (SaveFileDialog dialog = new SaveFileDialog())
            {
                dialog.Title = "Export evaluation report";
                dialog.Filter = "Text report (*.txt)|*.txt";
                dialog.FileName = "evaluation_report.txt";
                dialog.InitialDirectory = OutputFiles.GetOutputFolder();
                if (dialog.ShowDialog(this) != DialogResult.OK) return;

                try
                {
                    File.WriteAllText(dialog.FileName, BuildReportText(), Encoding.UTF8);
                    SetStatus("Report saved to " + dialog.FileName);
                }
                catch (Exception ex)
                {
                    ShowError("The report could not be saved", ex);
                }
            }
        }

        private string BuildReportText()
        {
            CultureInfo inv = CultureInfo.InvariantCulture;
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("MicroID - evaluation report");
            sb.AppendLine("Created: " + DateTime.Now.ToString("yyyy-MM-dd HH:mm", inv));
            sb.AppendLine();
            sb.AppendLine("MODEL");
            sb.AppendLine(lastSettingsText);
            sb.AppendLine("Classes: " + string.Join(", ", model.ClassNames));
            sb.AppendLine();

            if (dataset != null && model.IsCompatibleWith(dataset))
            {
                sb.AppendLine("DATA");
                sb.AppendLine("Images: " + dataset.Samples.Count + "   training: " + trainSet.Count + "   test: " + testSet.Count +
                              "   (stratified split, test = " + numTestPercent.Value.ToString("0", inv) + "%)");
                int[] all = dataset.CountPerClass(dataset.Samples);
                int[] tr = dataset.CountPerClass(trainSet);
                int[] te = dataset.CountPerClass(testSet);
                for (int c = 0; c < dataset.ClassNames.Count; c++)
                    sb.AppendLine("  " + dataset.ClassNames[c].PadRight(24) + all[c] + " images (" + tr[c] + " train / " + te[c] + " test)");
                sb.AppendLine("Pixels are converted to grey, resized to " + dataset.ImageSide + "x" + dataset.ImageSide +
                              " and MinMax-scaled to [0,1] (value / 255).");
                sb.AppendLine();
            }

            if (lastTestMetrics != null) sb.AppendLine(lastTestMetrics.ToText(model.ClassNames, "TEST SET RESULTS"));
            if (lastTrainMetrics != null)
            {
                sb.AppendLine();
                sb.AppendLine(lastTrainMetrics.ToText(model.ClassNames, "TRAINING SET RESULTS"));
            }

            if (history.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("TRAINING HISTORY (loss = mean squared error)");
                foreach (EpochResult r in history) sb.AppendLine(FormatEpoch(r));
            }
            return sb.ToString();
        }

        private async void SaveWeights_Click(object sender, EventArgs e)
        {
            if (model == null)
            {
                Info("There is no network to save yet.");
                return;
            }

            using (SaveFileDialog dialog = new SaveFileDialog())
            {
                dialog.Title = "Save the network weights";
                dialog.Filter = "Weights file (*.weights)|*.weights";
                dialog.DefaultExt = "weights";
                dialog.FileName = "microorganisms.weights";
                dialog.InitialDirectory = OutputFiles.GetOutputFolder();
                if (dialog.ShowDialog(this) != DialogResult.OK) return;

                string path = dialog.FileName;
                ClassifierModel net = model;
                SetIndeterminate(true);
                SetBusy(true, "Saving weights...");
                try
                {
                    await Task.Run(() => net.Save(path));
                    SetStatus("Weights saved to " + path + "  (keep the .meta file next to it).");
                }
                catch (Exception ex)
                {
                    ShowError("The weights could not be saved", ex);
                }
                finally
                {
                    SetIndeterminate(false);
                    SetBusy(false, null);
                }
            }
        }

        private async void LoadWeights_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Title = "Load network weights";
                dialog.Filter = "Weights file (*.weights)|*.weights|All files (*.*)|*.*";
                dialog.InitialDirectory = OutputFiles.GetOutputFolder();
                if (dialog.ShowDialog(this) != DialogResult.OK) return;

                string path = dialog.FileName;
                SetIndeterminate(true);
                SetBusy(true, "Loading weights...");
                try
                {
                    ClassifierModel loaded = await Task.Run(() => ClassifierModel.Load(path));
                    model = loaded;
                    history.Clear();
                    lastTrainMetrics = null;
                    lastTestMetrics = null;
                    lastSettingsText = "Network " + loaded.InputCount + " -> " + loaded.HiddenCount + " -> " +
                        loaded.OutputCount + " loaded from " + Path.GetFileName(path) + ".";
                    ResetTrainingViews();
                    ResetEvaluationViews();
                    UpdateArchitectureLabel();

                    if (loaded.HiddenCount >= numHidden.Minimum && loaded.HiddenCount <= numHidden.Maximum)
                        numHidden.Value = loaded.HiddenCount;
                    if ((decimal)loaded.LearningRateOutput >= numLrOutput.Minimum && (decimal)loaded.LearningRateOutput <= numLrOutput.Maximum)
                        numLrOutput.Value = (decimal)loaded.LearningRateOutput;
                    if ((decimal)loaded.LearningRateInput >= numLrInput.Minimum && (decimal)loaded.LearningRateInput <= numLrInput.Maximum)
                        numLrInput.Value = (decimal)loaded.LearningRateInput;
                    int sizeIndex = Array.IndexOf(ImageSizes, loaded.ImageSide);
                    if (sizeIndex >= 0) cmbImageSize.SelectedIndex = sizeIndex;

                    LogLine("Weights loaded from " + path);
                    if (dataset != null && !loaded.IsCompatibleWith(dataset))
                        SetStatus("Weights loaded. They need " + loaded.ImageSide + "x" + loaded.ImageSide +
                                  " images and the classes listed in step 2; reload your images to evaluate.");
                    else
                        SetStatus("Weights loaded. You can now evaluate or identify images.");
                }
                catch (Exception ex)
                {
                    ShowError("The weights could not be loaded", ex);
                }
                finally
                {
                    SetIndeterminate(false);
                    SetBusy(false, null);
                }
            }
        }

        // =====================================================================
        // Identify a single image
        // =====================================================================

        private void Identify_Click(object sender, EventArgs e)
        {
            if (model == null)
            {
                Info("Create and train a network (or load saved weights) first.");
                return;
            }

            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Title = "Open an image to identify";
                dialog.Filter = "Images (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png";
                if (dialog.ShowDialog(this) != DialogResult.OK) return;

                try
                {
                    string path = dialog.FileName;
                    double[] pixels = ImageDataset.LoadPixels(path, model.ImageSide);
                    double[] outputs = model.Predict(pixels);
                    int best = ClassifierModel.ArgMax(outputs);

                    using (Bitmap temp = new Bitmap(path))
                    {
                        Image old = picIdentify.Image;
                        picIdentify.Image = new Bitmap(temp);
                        if (old != null) old.Dispose();
                    }

                    barsPrediction.SetValues(model.ClassNames.ToArray(), outputs);
                    lblPrediction.Text = model.ClassNames[best] + "   (" +
                        (outputs[best] * 100.0).ToString("0.0", CultureInfo.InvariantCulture) + "%)";

                    string expected = ImageDataset.ExtractClassName(path);
                    int expectedIndex = model.ClassNames.FindIndex(
                        c => string.Equals(c, expected, StringComparison.OrdinalIgnoreCase));
                    StringBuilder info = new StringBuilder();
                    info.AppendLine(Path.GetFileName(path));
                    if (expectedIndex >= 0)
                        info.AppendLine(expectedIndex == best
                            ? "The file name says \"" + model.ClassNames[expectedIndex] + "\": correct."
                            : "The file name says \"" + model.ClassNames[expectedIndex] + "\": the network got it wrong.");
                    else
                        info.AppendLine("The file name does not match a known class.");
                    if (outputs[best] < 0.5)
                        info.AppendLine("Low confidence: no output neuron is above 50%.");
                    lblTruth.Text = info.ToString().TrimEnd();

                    SetStatus("Identified " + Path.GetFileName(path) + ".");
                }
                catch (Exception ex)
                {
                    ShowError("The image could not be identified", ex);
                }
            }
        }

        // =====================================================================
        // Hyper-parameter experiments
        // =====================================================================

        private async void RunExperiments_Click(object sender, EventArgs e)
        {
            if (dataset == null)
            {
                Info("Load your images first (step 1).");
                return;
            }
            if (testSet.Count == 0 || trainSet.Count == 0)
            {
                Info("Experiments need both training and test images.");
                return;
            }

            int[] hiddenSizes;
            double[] multipliers;
            double[] learningRates;
            int[] epochList;
            try
            {
                hiddenSizes = ParseIntList(txtExpHidden.Text, 1, 1024, "hidden neurons");
                multipliers = ParseDoubleList(txtExpMultipliers.Text, 0.01, 10.0, "weight multipliers");
                learningRates = ParseDoubleList(txtExpLearningRates.Text, 0.001, 2.0, "learning rates");
                epochList = ParseIntList(txtExpEpochs.Text, 1, 5000, "epochs");
            }
            catch (FormatException ex)
            {
                Info(ex.Message);
                return;
            }

            int maxEpoch = epochList.Max();
            int trainingRuns = hiddenSizes.Length * multipliers.Length * learningRates.Length;
            int totalEpochs = trainingRuns * maxEpoch;
            if (totalEpochs > 300)
            {
                DialogResult answer = MessageBox.Show(this,
                    trainingRuns + " networks will be trained for up to " + maxEpoch + " epochs each (" +
                    totalEpochs + " epochs in total). This can take a while - you can press Stop at any time.\n\nContinue?",
                    "MicroID", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                if (answer != DialogResult.Yes) return;
            }

            ImageDataset ds = dataset;
            List<ImageSample> train = trainSet;
            List<ImageSample> test = testSet;
            int seed = (int)numSeed.Value;
            // The input-layer rate keeps the ratio set in step 2 (default 0.15 / 0.2 = 0.75).
            double inputRatio = (double)(numLrInput.Value / numLrOutput.Value);

            experimentResults.Clear();
            gridExperiments.Rows.Clear();
            bestExperiment = null;
            lblExpBest.Text = "Running " + trainingRuns + " training run(s)...";

            CancellationTokenSource cts = new CancellationTokenSource();
            activeCts = cts;
            progressBar.Style = ProgressBarStyle.Blocks;
            progressBar.Minimum = 0;
            progressBar.Maximum = totalEpochs;
            progressBar.Value = 0;
            SetBusy(true, "Running experiments...");

            try
            {
                Action<ExperimentResult> onResult = delegate (ExperimentResult r)
                {
                    InvokeOnUi(delegate { AddExperimentRow(r); });
                };
                Action<int, int> onStep = delegate (int step, int total)
                {
                    InvokeOnUi(delegate
                    {
                        progressBar.Value = Math.Min(step, progressBar.Maximum);
                        SetStatus("Experiments: epoch " + step + " of " + total);
                    });
                };

                List<ExperimentResult> results = await Task.Run(() =>
                    HyperparameterExperiment.Run(ds, train, test, hiddenSizes, multipliers, learningRates, inputRatio,
                                                 epochList, seed, onResult, onStep, cts.Token));

                bool stopped = cts.IsCancellationRequested;
                bestExperiment = HyperparameterExperiment.FindBest(results);

                if (results.Count > 0)
                {
                    string path = OutputFiles.NewFilePath("hyperparameter_experiments", ".csv");
                    OutputFiles.WriteExperimentCsv(path, results);
                    HighlightBestExperiment();
                    LogLine("Experiment table saved to: " + path);
                    SetStatus((stopped ? "Experiments stopped. " : "Experiments finished. ") + "Results saved to " + path);
                }
                else
                {
                    lblExpBest.Text = "No finished runs.";
                    SetStatus("Experiments stopped before the first result.");
                }
            }
            catch (Exception ex)
            {
                ShowError("The experiments failed", ex);
            }
            finally
            {
                activeCts = null;
                cts.Dispose();
                SetBusy(false, null);
            }
        }

        private void AddExperimentRow(ExperimentResult r)
        {
            CultureInfo inv = CultureInfo.InvariantCulture;
            experimentResults.Add(r);
            gridExperiments.Rows.Add(
                r.Run,
                r.Hidden,
                r.InitMultiplier.ToString("0.##", inv),
                r.LearningRateOutput.ToString("0.####", inv),
                r.LearningRateInput.ToString("0.####", inv),
                r.Epochs,
                r.TrainLoss.ToString("0.0000", inv),
                (r.TrainAccuracy * 100.0).ToString("0.0", inv) + "%",
                r.TestLoss.ToString("0.0000", inv),
                (r.TestAccuracy * 100.0).ToString("0.0", inv) + "%",
                r.Seconds.ToString("0.0", inv));
        }

        private void HighlightBestExperiment()
        {
            if (bestExperiment == null) return;
            CultureInfo inv = CultureInfo.InvariantCulture;

            foreach (DataGridViewRow row in gridExperiments.Rows)
            {
                if (row.Cells[0].Value != null && (int)row.Cells[0].Value == bestExperiment.Run)
                {
                    row.DefaultCellStyle.BackColor = Color.FromArgb(214, 243, 222);
                    row.DefaultCellStyle.Font = new Font("Segoe UI Semibold", 9.5f);
                }
            }
            lblExpBest.Text = "Best: " + bestExperiment.Hidden + " hidden neurons, weight multiplier " +
                bestExperiment.InitMultiplier.ToString("0.##", inv) + ", learning rates " +
                bestExperiment.LearningRateOutput.ToString("0.####", inv) + " / " +
                bestExperiment.LearningRateInput.ToString("0.####", inv) + ", " + bestExperiment.Epochs +
                " epochs  ->  test accuracy " + (bestExperiment.TestAccuracy * 100.0).ToString("0.0", inv) +
                "%, test loss " + bestExperiment.TestLoss.ToString("0.0000", inv);
        }

        private void ApplyBest_Click(object sender, EventArgs e)
        {
            if (bestExperiment == null) return;
            numHidden.Value = Math.Max(numHidden.Minimum, Math.Min(numHidden.Maximum, bestExperiment.Hidden));
            numInitMultiplier.Value = Math.Max(numInitMultiplier.Minimum,
                Math.Min(numInitMultiplier.Maximum, (decimal)bestExperiment.InitMultiplier));
            numLrOutput.Value = Math.Max(numLrOutput.Minimum,
                Math.Min(numLrOutput.Maximum, (decimal)bestExperiment.LearningRateOutput));
            numLrInput.Value = Math.Max(numLrInput.Minimum,
                Math.Min(numLrInput.Maximum, (decimal)bestExperiment.LearningRateInput));
            numEpochs.Value = Math.Max(numEpochs.Minimum, Math.Min(numEpochs.Maximum, bestExperiment.Epochs));
            SetStatus("Best settings copied. Press 'Create Network', then 'Train' to use them.");
        }

        private static int[] ParseIntList(string text, int min, int max, string what)
        {
            List<int> values = new List<int>();
            foreach (string part in text.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries))
            {
                int v;
                if (!int.TryParse(part.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out v) || v < min || v > max)
                    throw new FormatException("'" + part + "' is not valid for " + what + ". Use whole numbers from " +
                        min + " to " + max + ", separated by commas.");
                if (!values.Contains(v)) values.Add(v);
            }
            if (values.Count == 0) throw new FormatException("Enter at least one value for " + what + ".");
            return values.ToArray();
        }

        private static double[] ParseDoubleList(string text, double min, double max, string what)
        {
            List<double> values = new List<double>();
            foreach (string part in text.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries))
            {
                double v;
                if (!double.TryParse(part.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out v) || v < min || v > max)
                    throw new FormatException("'" + part + "' is not valid for " + what + ". Use numbers from " +
                        min.ToString(CultureInfo.InvariantCulture) + " to " + max.ToString(CultureInfo.InvariantCulture) +
                        ", with a dot as decimal separator, separated by commas.");
                if (!values.Contains(v)) values.Add(v);
            }
            if (values.Count == 0) throw new FormatException("Enter at least one value for " + what + ".");
            return values.ToArray();
        }

        // =====================================================================
        // Shared helpers
        // =====================================================================

        private bool EnsureModelMatchesDataset()
        {
            if (dataset == null)
            {
                Info("Load your images first (step 1).");
                return false;
            }
            if (model == null)
            {
                Info("Create the network first (step 2), or load saved weights.");
                return false;
            }
            if (!model.IsCompatibleWith(dataset))
            {
                Info("The network does not match the loaded images (different image size or classes).\n\n" +
                     "Create a new network (step 2), or reload your images with the same size and classes.");
                return false;
            }
            return true;
        }

        private void SetBusy(bool busy, string status)
        {
            isBusy = busy;
            foreach (Control c in lockableControls) c.Enabled = !busy;
            btnStop.Enabled = busy;
            btnStopExperiments.Enabled = busy;
            if (!busy) btnApplyBest.Enabled = bestExperiment != null;
            if (status != null) SetStatus(status);

            if (!busy && closeRequested)
            {
                closeRequested = false;
                BeginInvoke(new Action(Close));
            }
        }

        private void SetIndeterminate(bool on)
        {
            if (on)
            {
                progressBar.Style = ProgressBarStyle.Marquee;
                progressBar.MarqueeAnimationSpeed = 30;
            }
            else
            {
                progressBar.Style = ProgressBarStyle.Blocks;
                progressBar.Value = 0;
            }
        }

        private void SetStatus(string text)
        {
            lblStatus.Text = text;
        }

        private void LogLine(string text)
        {
            txtLog.AppendText(text + Environment.NewLine);
        }

        private void Info(string text)
        {
            MessageBox.Show(this, text, "MicroID", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void ShowError(string title, Exception ex)
        {
            MessageBox.Show(this, title + ":\n\n" + ex.Message, "MicroID", MessageBoxButtons.OK, MessageBoxIcon.Error);
            SetStatus(title + ".");
        }

        /// <summary>Runs an action on the UI thread and waits for it (used by the training thread).</summary>
        private void InvokeOnUi(Action action)
        {
            try
            {
                if (IsDisposed || !IsHandleCreated) return;
                Invoke(action);
            }
            catch (ObjectDisposedException) { }
            catch (InvalidOperationException) { }
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (!isBusy) return;

            DialogResult answer = MessageBox.Show(this,
                "A task is still running. Stop it and close the program?", "MicroID",
                MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (answer != DialogResult.Yes)
            {
                e.Cancel = true;
                return;
            }

            // Cancel the task and close as soon as it has wound down (see SetBusy).
            e.Cancel = true;
            closeRequested = true;
            if (activeCts != null) activeCts.Cancel();
            SetStatus("Stopping, the program will close in a moment...");
        }
    }
}
