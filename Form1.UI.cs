using System;
using System.Drawing;
using System.Windows.Forms;

namespace BackPropNNImages
{
    /// <summary>
    /// Form1 (part 2): builds the whole user interface in code - header, step cards on the left,
    /// result tabs on the right, status bar. The program logic lives in Form1.cs.
    /// </summary>
    public partial class Form1
    {
        // ---- colour palette -------------------------------------------------
        private static readonly Color Navy = Color.FromArgb(18, 38, 63);
        private static readonly Color Accent = Color.FromArgb(37, 99, 235);
        private static readonly Color AccentDark = Color.FromArgb(29, 78, 216);
        private static readonly Color PageBack = Color.FromArgb(244, 246, 250);
        private static readonly Color BorderColor = Color.FromArgb(222, 227, 235);
        private static readonly Color TextDark = Color.FromArgb(25, 40, 60);
        private static readonly Color TextMuted = Color.FromArgb(100, 112, 130);
        private static readonly Color Danger = Color.FromArgb(214, 48, 64);
        private static readonly Color Orange = Color.FromArgb(234, 128, 28);

        /// <summary>Selectable image sizes (the network gets side x side inputs).</summary>
        private static readonly int[] ImageSizes = { 16, 24, 32, 48, 64, 100 };

        private enum ButtonKind { Primary, Secondary, Danger }

        // ---- header / status ------------------------------------------------
        private Panel headerPanel;
        private StatusStrip statusStrip;
        private ToolStripStatusLabel lblStatus;
        private ToolStripProgressBar progressBar;

        // ---- step cards (left) ----------------------------------------------
        private Button btnLoadFolder, btnLoadFiles;
        private NumericUpDown numTestPercent;
        private ComboBox cmbImageSize;
        private Label lblDatasetSummary;

        private NumericUpDown numHidden, numInitMultiplier, numLrOutput, numLrInput, numSeed;
        private Button btnCreate;
        private Label lblArchitecture;

        private NumericUpDown numEpochs, numLogEvery;
        private Button btnTrain, btnStop, btnOpenOutputs;

        private Button btnEvaluate, btnExport, btnSaveWeights, btnLoadWeights;

        // ---- tabs (right) ---------------------------------------------------
        private TabControl tabs;
        private TabPage tabDataset, tabTraining, tabEvaluation, tabExperiments, tabIdentify;

        private FlowLayoutPanel flowThumbs;
        private DataGridView gridClasses;
        private TextBox txtSkipped;

        private LineChart chartLoss, chartAccuracy;
        private int lossTrainSeries, lossTestSeries, accTrainSeries, accTestSeries;
        private TextBox txtLog;

        private Label lblEvalSummary;
        private DataGridView gridConfusion, gridMetrics;

        private TextBox txtExpHidden, txtExpMultipliers, txtExpLearningRates, txtExpEpochs;
        private Button btnRunExperiments, btnStopExperiments, btnApplyBest;
        private Label lblExpBest;
        private DataGridView gridExperiments;

        private PictureBox picIdentify;
        private Button btnOpenImage;
        private Label lblPrediction, lblTruth;
        private ProbabilityBars barsPrediction;

        // =====================================================================
        // Top level
        // =====================================================================

        private void BuildUi()
        {
            SuspendLayout();

            Font = new Font("Segoe UI", 9.5f);
            BackColor = PageBack;
            MinimumSize = new Size(1080, 720);
            Size = new Size(1280, 820);

            // Controls are docked in reverse order of adding: the Fill control first, edge panels after.
            Panel rightPanel = new Panel();
            rightPanel.Dock = DockStyle.Fill;
            rightPanel.Padding = new Padding(0, 12, 12, 12);
            rightPanel.Controls.Add(BuildTabs());
            Controls.Add(rightPanel);

            Controls.Add(BuildLeftPanel());
            Controls.Add(BuildStatusStrip());
            Controls.Add(BuildHeader());

            ResumeLayout(true);
        }

        private Control BuildHeader()
        {
            headerPanel = new Panel();
            headerPanel.Dock = DockStyle.Top;
            headerPanel.Height = 66;
            headerPanel.BackColor = Navy;

            Label title = new Label();
            title.Text = "MicroID";
            title.Font = new Font("Segoe UI Semibold", 20f);
            title.ForeColor = Color.White;
            title.AutoSize = true;
            title.Location = new Point(18, 8);

            Label subtitle = new Label();
            subtitle.Text = "Microorganism classifier  |  3-layer back-propagation neural network";
            subtitle.Font = new Font("Segoe UI", 10f);
            subtitle.ForeColor = Color.FromArgb(170, 190, 215);
            subtitle.AutoSize = true;
            subtitle.Location = new Point(150, 24);

            headerPanel.Controls.Add(title);
            headerPanel.Controls.Add(subtitle);
            return headerPanel;
        }

        private Control BuildStatusStrip()
        {
            statusStrip = new StatusStrip();
            statusStrip.BackColor = Color.White;
            statusStrip.SizingGrip = false;

            lblStatus = new ToolStripStatusLabel("Ready. Step 1: load your images.");
            lblStatus.Spring = true;
            lblStatus.TextAlign = ContentAlignment.MiddleLeft;
            lblStatus.ForeColor = TextDark;

            progressBar = new ToolStripProgressBar();
            progressBar.AutoSize = false;
            progressBar.Width = 240;
            progressBar.Minimum = 0;
            progressBar.Maximum = 100;

            statusStrip.Items.Add(lblStatus);
            statusStrip.Items.Add(progressBar);
            return statusStrip;
        }

        // =====================================================================
        // Left side: the four step cards
        // =====================================================================

        private Control BuildLeftPanel()
        {
            Panel scroll = new Panel();
            scroll.Dock = DockStyle.Left;
            scroll.Width = 392;
            scroll.AutoScroll = true;
            scroll.BackColor = PageBack;
            scroll.Padding = new Padding(12, 12, 6, 12);

            TableLayoutPanel stack = new TableLayoutPanel();
            stack.Dock = DockStyle.Top;
            stack.AutoSize = true;
            stack.AutoSizeMode = AutoSizeMode.GrowAndShrink;
            stack.ColumnCount = 1;
            stack.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));

            // ---- Card 1: images -------------------------------------------
            Card c1 = new Card("1   Load images");
            btnLoadFolder = MakeButton("Open Folder...", ButtonKind.Primary);
            btnLoadFiles = MakeButton("Select Images...", ButtonKind.Secondary);
            c1.Controls.Add(ButtonRow(btnLoadFolder, btnLoadFiles));

            numTestPercent = MakeNumeric(5, 50, 20, 0, 5);
            c1.Controls.Add(LabeledRow("Test split (%)", numTestPercent));

            cmbImageSize = new ComboBox();
            cmbImageSize.DropDownStyle = ComboBoxStyle.DropDownList;
            foreach (int side in ImageSizes)
                cmbImageSize.Items.Add(side + " x " + side + "  (" + (side * side).ToString("N0") + " inputs)");
            cmbImageSize.SelectedIndex = Array.IndexOf(ImageSizes, 32);
            c1.Controls.Add(LabeledRow("Image size", cmbImageSize));

            lblDatasetSummary = MakeInfoLabel("No images loaded yet. File names must look like  name (1).png");
            c1.Controls.Add(lblDatasetSummary);
            stack.Controls.Add(c1);

            // ---- Card 2: network ------------------------------------------
            Card c2 = new Card("2   Create the network");
            numHidden = MakeNumeric(1, 1024, 64, 0, 8);
            c2.Controls.Add(LabeledRow("Hidden neurons", numHidden));
            numInitMultiplier = MakeNumeric(0.1m, 10m, 1.0m, 1, 0.1m);
            c2.Controls.Add(LabeledRow("Initial weight multiplier", numInitMultiplier));
            numLrOutput = MakeNumeric(0.001m, 2m, 0.2m, 3, 0.01m);
            c2.Controls.Add(LabeledRow("Learning rate (output layer)", numLrOutput));
            numLrInput = MakeNumeric(0.001m, 2m, 0.15m, 3, 0.01m);
            c2.Controls.Add(LabeledRow("Learning rate (input layer)", numLrInput));
            numSeed = MakeNumeric(0, 99999, 42, 0, 1);
            c2.Controls.Add(LabeledRow("Random seed", numSeed));
            btnCreate = MakeButton("Create Network", ButtonKind.Primary);
            c2.Controls.Add(ButtonRow(btnCreate));
            lblArchitecture = MakeInfoLabel("No network yet.");
            c2.Controls.Add(lblArchitecture);
            stack.Controls.Add(c2);

            // ---- Card 3: training -----------------------------------------
            Card c3 = new Card("3   Train");
            numEpochs = MakeNumeric(1, 5000, 30, 0, 10);
            c3.Controls.Add(LabeledRow("Epochs", numEpochs));
            numLogEvery = MakeNumeric(1, 1000, 1, 0, 1);
            c3.Controls.Add(LabeledRow("Show in log every N epochs", numLogEvery));
            btnTrain = MakeButton("Train", ButtonKind.Primary);
            btnStop = MakeButton("Stop", ButtonKind.Danger);
            btnStop.Enabled = false;
            c3.Controls.Add(ButtonRow(btnTrain, btnStop));
            btnOpenOutputs = MakeButton("Open Output Folder", ButtonKind.Secondary);
            c3.Controls.Add(ButtonRow(btnOpenOutputs));
            c3.Controls.Add(MakeInfoLabel("Every epoch is also written to a CSV file in the 'outputs' folder."));
            stack.Controls.Add(c3);

            // ---- Card 4: evaluate / save ----------------------------------
            Card c4 = new Card("4   Evaluate and save");
            btnEvaluate = MakeButton("Evaluate Test Set", ButtonKind.Primary);
            btnExport = MakeButton("Export Report...", ButtonKind.Secondary);
            c4.Controls.Add(ButtonRow(btnEvaluate, btnExport));
            btnSaveWeights = MakeButton("Save Weights...", ButtonKind.Secondary);
            btnLoadWeights = MakeButton("Load Weights...", ButtonKind.Secondary);
            c4.Controls.Add(ButtonRow(btnSaveWeights, btnLoadWeights));
            stack.Controls.Add(c4);

            scroll.Controls.Add(stack);
            return scroll;
        }

        // =====================================================================
        // Right side: tabs
        // =====================================================================

        private Control BuildTabs()
        {
            tabs = new TabControl();
            tabs.Dock = DockStyle.Fill;
            tabs.Padding = new Point(18, 6);
            tabs.Font = new Font("Segoe UI Semibold", 10f);

            tabDataset = NewTab("Dataset");
            tabTraining = NewTab("Training");
            tabEvaluation = NewTab("Evaluation");
            tabExperiments = NewTab("Experiments");
            tabIdentify = NewTab("Identify");

            BuildDatasetTab();
            BuildTrainingTab();
            BuildEvaluationTab();
            BuildExperimentsTab();
            BuildIdentifyTab();

            tabs.TabPages.Add(tabDataset);
            tabs.TabPages.Add(tabTraining);
            tabs.TabPages.Add(tabEvaluation);
            tabs.TabPages.Add(tabExperiments);
            tabs.TabPages.Add(tabIdentify);
            return tabs;
        }

        private TabPage NewTab(string title)
        {
            TabPage page = new TabPage(title);
            page.UseVisualStyleBackColor = false;
            page.BackColor = PageBack;
            page.Padding = new Padding(12);
            page.Font = new Font("Segoe UI", 9.5f);
            return page;
        }

        private TableLayoutPanel NewLayout(int columns)
        {
            TableLayoutPanel t = new TableLayoutPanel();
            t.Dock = DockStyle.Fill;
            t.ColumnCount = columns;
            t.BackColor = PageBack;
            return t;
        }

        // ---- Dataset tab -------------------------------------------------------

        private void BuildDatasetTab()
        {
            TableLayoutPanel t = NewLayout(1);
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, 26f));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, 146f));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, 26f));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, 26f));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, 66f));

            t.Controls.Add(MakeCaption("One example image per class (after resizing to the network input size)"), 0, 0);

            flowThumbs = new FlowLayoutPanel();
            flowThumbs.Dock = DockStyle.Fill;
            flowThumbs.BackColor = Color.White;
            flowThumbs.AutoScroll = true;
            flowThumbs.Padding = new Padding(6);
            t.Controls.Add(Wrap(flowThumbs), 0, 1);

            t.Controls.Add(MakeCaption("Images per class"), 0, 2);

            gridClasses = MakeGrid();
            AddColumn(gridClasses, "Class", 40, false);
            AddColumn(gridClasses, "Images", 15, true);
            AddColumn(gridClasses, "Training", 15, true);
            AddColumn(gridClasses, "Test", 15, true);
            t.Controls.Add(Wrap(gridClasses), 0, 3);

            t.Controls.Add(MakeCaption("Files that could not be used"), 0, 4);

            txtSkipped = new TextBox();
            txtSkipped.Multiline = true;
            txtSkipped.ReadOnly = true;
            txtSkipped.ScrollBars = ScrollBars.Vertical;
            txtSkipped.BorderStyle = BorderStyle.None;
            txtSkipped.BackColor = Color.White;
            txtSkipped.Dock = DockStyle.Fill;
            txtSkipped.Text = "None";
            t.Controls.Add(Wrap(txtSkipped), 0, 5);

            tabDataset.Controls.Add(t);
        }

        // ---- Training tab ------------------------------------------------------

        private void BuildTrainingTab()
        {
            TableLayoutPanel t = NewLayout(2);
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50f));
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50f));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 58f));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 42f));

            chartLoss = new LineChart();
            chartLoss.Title = "Loss (mean squared error)";
            chartLoss.Dock = DockStyle.Fill;
            lossTrainSeries = chartLoss.AddSeries("training", Accent);
            lossTestSeries = chartLoss.AddSeries("test", Orange);
            Control wrapLoss = Wrap(chartLoss);
            wrapLoss.Margin = new Padding(0, 0, 6, 8);
            t.Controls.Add(wrapLoss, 0, 0);

            chartAccuracy = new LineChart();
            chartAccuracy.Title = "Accuracy";
            chartAccuracy.FixedUnitRange = true;
            chartAccuracy.YFormat = "0%";
            chartAccuracy.Dock = DockStyle.Fill;
            accTrainSeries = chartAccuracy.AddSeries("training", Accent);
            accTestSeries = chartAccuracy.AddSeries("test", Orange);
            Control wrapAcc = Wrap(chartAccuracy);
            wrapAcc.Margin = new Padding(6, 0, 0, 8);
            t.Controls.Add(wrapAcc, 1, 0);

            txtLog = new TextBox();
            txtLog.Multiline = true;
            txtLog.ReadOnly = true;
            txtLog.ScrollBars = ScrollBars.Vertical;
            txtLog.BorderStyle = BorderStyle.None;
            txtLog.BackColor = Color.White;
            txtLog.Font = new Font("Consolas", 9.5f);
            txtLog.Dock = DockStyle.Fill;
            Control wrapLog = Wrap(txtLog);
            t.Controls.Add(wrapLog, 0, 1);
            t.SetColumnSpan(wrapLog, 2);

            tabTraining.Controls.Add(t);
        }

        // ---- Evaluation tab ----------------------------------------------------

        private void BuildEvaluationTab()
        {
            TableLayoutPanel t = NewLayout(1);
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, 58f));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, 26f));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 56f));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, 26f));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 44f));

            lblEvalSummary = new Label();
            lblEvalSummary.Dock = DockStyle.Fill;
            lblEvalSummary.Font = new Font("Segoe UI Semibold", 12f);
            lblEvalSummary.ForeColor = TextDark;
            lblEvalSummary.TextAlign = ContentAlignment.MiddleLeft;
            lblEvalSummary.Text = "No evaluation yet. Train the network or press 'Evaluate Test Set'.";
            t.Controls.Add(lblEvalSummary, 0, 0);

            t.Controls.Add(MakeCaption("Confusion matrix (rows = actual class, columns = predicted class)"), 0, 1);
            gridConfusion = MakeGrid();
            t.Controls.Add(Wrap(gridConfusion), 0, 2);

            t.Controls.Add(MakeCaption("Per-class precision, recall and F1"), 0, 3);
            gridMetrics = MakeGrid();
            AddColumn(gridMetrics, "Class", 34, false);
            AddColumn(gridMetrics, "Test images", 16, true);
            AddColumn(gridMetrics, "Precision", 16, true);
            AddColumn(gridMetrics, "Recall", 16, true);
            AddColumn(gridMetrics, "F1 score", 16, true);
            t.Controls.Add(Wrap(gridMetrics), 0, 4);

            tabEvaluation.Controls.Add(t);
        }

        // ---- Experiments tab ---------------------------------------------------

        private void BuildExperimentsTab()
        {
            TableLayoutPanel t = NewLayout(1);
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, 26f));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, 140f));
            t.RowStyles.Add(new RowStyle(SizeType.Absolute, 34f));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));

            t.Controls.Add(MakeCaption("Hyper-parameter experiments: every combination is trained and compared on the same split"), 0, 0);

            Card form = new Card("Values to try (comma separated)");
            form.Margin = new Padding(0);
            form.Dock = DockStyle.Fill;

            txtExpHidden = new TextBox();
            txtExpHidden.Text = "16, 32, 64, 128";
            txtExpMultipliers = new TextBox();
            txtExpMultipliers.Text = "0.5, 1, 2";
            txtExpLearningRates = new TextBox();
            txtExpLearningRates.Text = "0.05, 0.1, 0.2";
            txtExpEpochs = new TextBox();
            txtExpEpochs.Text = "10, 20, 40";

            TableLayoutPanel inputs = new TableLayoutPanel();
            inputs.ColumnCount = 8;
            inputs.RowCount = 1;
            inputs.Dock = DockStyle.Fill;
            inputs.Height = 34;
            for (int i = 0; i < 4; i++)
            {
                inputs.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
                inputs.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 25f));
            }
            inputs.Controls.Add(MakeInlineLabel("Hidden neurons"), 0, 0);
            AddInline(inputs, txtExpHidden, 1);
            inputs.Controls.Add(MakeInlineLabel("Weight multipliers"), 2, 0);
            AddInline(inputs, txtExpMultipliers, 3);
            inputs.Controls.Add(MakeInlineLabel("Learning rates"), 4, 0);
            AddInline(inputs, txtExpLearningRates, 5);
            inputs.Controls.Add(MakeInlineLabel("Epochs"), 6, 0);
            AddInline(inputs, txtExpEpochs, 7);
            form.Controls.Add(inputs);

            btnRunExperiments = MakeButton("Run Experiments", ButtonKind.Primary);
            btnStopExperiments = MakeButton("Stop", ButtonKind.Danger);
            btnStopExperiments.Enabled = false;
            btnApplyBest = MakeButton("Use Best Settings", ButtonKind.Secondary);
            btnApplyBest.Enabled = false;
            TableLayoutPanel buttons = ButtonRow(btnRunExperiments, btnStopExperiments, btnApplyBest);
            form.Controls.Add(buttons);
            t.Controls.Add(form, 0, 1);

            lblExpBest = new Label();
            lblExpBest.Dock = DockStyle.Fill;
            lblExpBest.TextAlign = ContentAlignment.MiddleLeft;
            lblExpBest.Font = new Font("Segoe UI Semibold", 10f);
            lblExpBest.ForeColor = TextDark;
            lblExpBest.Text = "No experiments run yet.";
            t.Controls.Add(lblExpBest, 0, 2);

            gridExperiments = MakeGrid();
            AddColumn(gridExperiments, "Run", 6, true);
            AddColumn(gridExperiments, "Hidden", 9, true);
            AddColumn(gridExperiments, "Weight x", 9, true);
            AddColumn(gridExperiments, "LR out", 9, true);
            AddColumn(gridExperiments, "LR in", 9, true);
            AddColumn(gridExperiments, "Epochs", 8, true);
            AddColumn(gridExperiments, "Train loss", 11, true);
            AddColumn(gridExperiments, "Train acc.", 10, true);
            AddColumn(gridExperiments, "Test loss", 11, true);
            AddColumn(gridExperiments, "Test acc.", 10, true);
            AddColumn(gridExperiments, "Seconds", 9, true);
            t.Controls.Add(Wrap(gridExperiments), 0, 3);

            tabExperiments.Controls.Add(t);
        }

        // ---- Identify tab ------------------------------------------------------

        private void BuildIdentifyTab()
        {
            TableLayoutPanel t = NewLayout(2);
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 300f));
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));
            t.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));

            // left: picture + button
            TableLayoutPanel left = new TableLayoutPanel();
            left.Dock = DockStyle.Fill;
            left.ColumnCount = 1;
            left.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));
            left.RowStyles.Add(new RowStyle(SizeType.Absolute, 286f));
            left.RowStyles.Add(new RowStyle(SizeType.Absolute, 52f));
            left.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));

            picIdentify = new PictureBox();
            picIdentify.Dock = DockStyle.Fill;
            picIdentify.SizeMode = PictureBoxSizeMode.Zoom;
            picIdentify.BackColor = Color.White;
            Control wrapPic = Wrap(picIdentify);
            wrapPic.Margin = new Padding(0, 0, 12, 0);
            left.Controls.Add(wrapPic, 0, 0);

            btnOpenImage = MakeButton("Open Image to Identify...", ButtonKind.Primary);
            TableLayoutPanel buttonRow = ButtonRow(btnOpenImage);
            buttonRow.Margin = new Padding(0, 8, 12, 0);
            left.Controls.Add(buttonRow, 0, 1);

            lblTruth = MakeInfoLabel("");
            lblTruth.MaximumSize = new Size(280, 0);
            left.Controls.Add(lblTruth, 0, 2);
            t.Controls.Add(left, 0, 0);

            // right: answer + bars
            TableLayoutPanel right = new TableLayoutPanel();
            right.Dock = DockStyle.Fill;
            right.ColumnCount = 1;
            right.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));
            right.RowStyles.Add(new RowStyle(SizeType.Absolute, 70f));
            right.RowStyles.Add(new RowStyle(SizeType.Absolute, 26f));
            right.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));

            lblPrediction = new Label();
            lblPrediction.Dock = DockStyle.Fill;
            lblPrediction.Font = new Font("Segoe UI Semibold", 18f);
            lblPrediction.ForeColor = TextDark;
            lblPrediction.TextAlign = ContentAlignment.MiddleLeft;
            lblPrediction.Text = "Load or train a network, then open an image.";
            right.Controls.Add(lblPrediction, 0, 0);

            right.Controls.Add(MakeCaption("Output neuron activations (one per class)"), 0, 1);

            barsPrediction = new ProbabilityBars();
            barsPrediction.Dock = DockStyle.Fill;
            right.Controls.Add(Wrap(barsPrediction), 0, 2);

            t.Controls.Add(right, 1, 0);
            tabIdentify.Controls.Add(t);
        }

        // =====================================================================
        // Small UI building blocks
        // =====================================================================

        /// <summary>A white panel with a thin border and a title; children stack vertically.</summary>
        private class Card : TableLayoutPanel
        {
            public Card(string title)
            {
                ColumnCount = 1;
                ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));
                AutoSize = true;
                AutoSizeMode = AutoSizeMode.GrowAndShrink;
                BackColor = Color.White;
                Padding = new Padding(14, 12, 14, 8);
                Margin = new Padding(0, 0, 0, 12);
                Dock = DockStyle.Fill;

                Label header = new Label();
                header.Text = title;
                header.Font = new Font("Segoe UI Semibold", 11f);
                header.ForeColor = TextDark;
                header.AutoSize = true;
                header.Margin = new Padding(0, 0, 0, 8);
                Controls.Add(header);
            }

            protected override void OnPaint(PaintEventArgs e)
            {
                base.OnPaint(e);
                using (Pen pen = new Pen(BorderColor))
                    e.Graphics.DrawRectangle(pen, 0, 0, Width - 1, Height - 1);
            }
        }

        private Button MakeButton(string text, ButtonKind kind)
        {
            Button b = new Button();
            b.Text = text;
            b.Tag = kind;
            b.Dock = DockStyle.Fill;
            b.Height = 36;
            b.Margin = new Padding(0, 4, 8, 4);
            b.Font = new Font("Segoe UI Semibold", 9.5f);
            b.FlatStyle = FlatStyle.Flat;
            b.UseVisualStyleBackColor = false;
            b.Cursor = Cursors.Hand;
            StyleButton(b);
            b.EnabledChanged += delegate { StyleButton(b); };
            return b;
        }

        private static void StyleButton(Button b)
        {
            ButtonKind kind = (ButtonKind)b.Tag;
            b.FlatAppearance.BorderSize = 1;

            if (!b.Enabled)
            {
                b.BackColor = Color.FromArgb(235, 238, 243);
                b.ForeColor = Color.FromArgb(150, 158, 170);
                b.FlatAppearance.BorderColor = BorderColor;
                b.Cursor = Cursors.Default;
                return;
            }

            b.Cursor = Cursors.Hand;
            switch (kind)
            {
                case ButtonKind.Primary:
                    b.BackColor = Accent;
                    b.ForeColor = Color.White;
                    b.FlatAppearance.BorderColor = Accent;
                    b.FlatAppearance.MouseOverBackColor = AccentDark;
                    break;
                case ButtonKind.Danger:
                    b.BackColor = Danger;
                    b.ForeColor = Color.White;
                    b.FlatAppearance.BorderColor = Danger;
                    b.FlatAppearance.MouseOverBackColor = Color.FromArgb(185, 35, 50);
                    break;
                default:
                    b.BackColor = Color.White;
                    b.ForeColor = TextDark;
                    b.FlatAppearance.BorderColor = Color.FromArgb(190, 198, 210);
                    b.FlatAppearance.MouseOverBackColor = Color.FromArgb(236, 240, 247);
                    break;
            }
        }

        /// <summary>Places buttons side by side with equal widths.</summary>
        private TableLayoutPanel ButtonRow(params Button[] buttons)
        {
            TableLayoutPanel row = new TableLayoutPanel();
            row.ColumnCount = buttons.Length;
            row.RowCount = 1;
            row.Dock = DockStyle.Fill;
            row.Height = 44;
            row.Margin = new Padding(0);
            for (int i = 0; i < buttons.Length; i++)
            {
                row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f / buttons.Length));
                row.Controls.Add(buttons[i], i, 0);
            }
            return row;
        }

        private Control LabeledRow(string text, Control input)
        {
            TableLayoutPanel row = new TableLayoutPanel();
            row.ColumnCount = 2;
            row.RowCount = 1;
            row.Dock = DockStyle.Fill;
            row.Height = 34;
            row.Margin = new Padding(0);
            row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 62f));
            row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 38f));

            Label label = new Label();
            label.Text = text;
            label.Dock = DockStyle.Fill;
            label.TextAlign = ContentAlignment.MiddleLeft;
            label.ForeColor = TextDark;

            input.Anchor = AnchorStyles.Left | AnchorStyles.Right;
            input.Margin = new Padding(0, 4, 8, 4);

            row.Controls.Add(label, 0, 0);
            row.Controls.Add(input, 1, 0);
            return row;
        }

        private static NumericUpDown MakeNumeric(decimal min, decimal max, decimal value, int decimals, decimal step)
        {
            NumericUpDown n = new NumericUpDown();
            n.Minimum = min;
            n.Maximum = max;
            n.DecimalPlaces = decimals;
            n.Increment = step;
            n.Value = value;
            n.TextAlign = HorizontalAlignment.Right;
            return n;
        }

        private static Label MakeInfoLabel(string text)
        {
            Label l = new Label();
            l.Text = text;
            l.AutoSize = true;
            l.MaximumSize = new Size(318, 0);
            l.ForeColor = TextMuted;
            l.Margin = new Padding(0, 6, 0, 4);
            return l;
        }

        private static Label MakeCaption(string text)
        {
            Label l = new Label();
            l.Text = text;
            l.Dock = DockStyle.Fill;
            l.TextAlign = ContentAlignment.BottomLeft;
            l.Font = new Font("Segoe UI Semibold", 9.5f);
            l.ForeColor = TextMuted;
            l.Padding = new Padding(0, 0, 0, 3);
            return l;
        }

        private static Label MakeInlineLabel(string text)
        {
            Label l = new Label();
            l.Text = text;
            l.AutoSize = true;
            l.Anchor = AnchorStyles.Left;
            l.Margin = new Padding(0, 0, 6, 0);
            l.ForeColor = TextDark;
            return l;
        }

        private static void AddInline(TableLayoutPanel panel, Control control, int column)
        {
            control.Anchor = AnchorStyles.Left | AnchorStyles.Right;
            control.Margin = new Padding(0, 0, 14, 0);
            panel.Controls.Add(control, column, 0);
        }

        /// <summary>Gives a control a 1 pixel border by placing it inside a coloured panel.</summary>
        private static Panel Wrap(Control inner)
        {
            Panel p = new Panel();
            p.Dock = DockStyle.Fill;
            p.BackColor = BorderColor;
            p.Padding = new Padding(1);
            p.Margin = new Padding(0);
            inner.Dock = DockStyle.Fill;
            p.Controls.Add(inner);
            return p;
        }

        private static DataGridView MakeGrid()
        {
            DataGridView g = new DataGridView();
            g.Dock = DockStyle.Fill;
            g.ReadOnly = true;
            g.AllowUserToAddRows = false;
            g.AllowUserToDeleteRows = false;
            g.AllowUserToResizeRows = false;
            g.RowHeadersVisible = false;
            g.BackgroundColor = Color.White;
            g.BorderStyle = BorderStyle.None;
            g.CellBorderStyle = DataGridViewCellBorderStyle.SingleHorizontal;
            g.GridColor = Color.FromArgb(234, 238, 244);
            g.EnableHeadersVisualStyles = false;
            g.ColumnHeadersBorderStyle = DataGridViewHeaderBorderStyle.None;
            g.ColumnHeadersDefaultCellStyle.BackColor = Color.FromArgb(240, 243, 248);
            g.ColumnHeadersDefaultCellStyle.ForeColor = TextDark;
            g.ColumnHeadersDefaultCellStyle.Font = new Font("Segoe UI Semibold", 9.5f);
            g.ColumnHeadersDefaultCellStyle.SelectionBackColor = Color.FromArgb(240, 243, 248);
            g.ColumnHeadersDefaultCellStyle.SelectionForeColor = TextDark;
            g.ColumnHeadersHeight = 32;
            g.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.DisableResizing;
            g.DefaultCellStyle.SelectionBackColor = Color.FromArgb(219, 232, 253);
            g.DefaultCellStyle.SelectionForeColor = TextDark;
            g.DefaultCellStyle.Padding = new Padding(4, 2, 4, 2);
            g.RowTemplate.Height = 28;
            g.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
            g.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
            g.MultiSelect = false;
            return g;
        }

        private static void AddColumn(DataGridView grid, string header, float fillWeight, bool alignRight)
        {
            DataGridViewTextBoxColumn column = new DataGridViewTextBoxColumn();
            column.HeaderText = header;
            column.FillWeight = fillWeight;
            column.SortMode = DataGridViewColumnSortMode.NotSortable;
            if (alignRight) column.DefaultCellStyle.Alignment = DataGridViewContentAlignment.MiddleRight;
            grid.Columns.Add(column);
        }
    }
}
