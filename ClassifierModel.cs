using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Backprop;

namespace BackPropNNImages
{
    /// <summary>Loss / accuracy of one finished training epoch.</summary>
    public class EpochResult
    {
        public int Epoch;

        /// <summary>Running (online) MSE over the training set during this epoch.</summary>
        public double TrainLoss;
        public double TrainAccuracy;

        /// <summary>MSE / accuracy on the held-out test set (NaN if there is no test set).</summary>
        public double TestLoss = double.NaN;
        public double TestAccuracy = double.NaN;

        public double ElapsedSeconds;
    }

    /// <summary>
    /// Thin wrapper around the teacher's Backprop.NeuralNet (input / hidden / output classes).
    /// The NeuralNet engine itself is NOT modified: this class only feeds it data,
    /// measures loss, and saves / loads its weights.
    ///
    /// One output neuron per class; target = one-hot vector; loss = mean squared error.
    /// </summary>
    public class ClassifierModel
    {
        private readonly NeuralNet net;

        public int ImageSide { get; private set; }
        public int HiddenCount { get; private set; }
        public List<string> ClassNames { get; private set; }

        /// <summary>Learning rate of the hidden-to-output weights (engine default 0.2).</summary>
        public double LearningRateOutput { get; private set; }

        /// <summary>Learning rate of the input-to-hidden weights (engine default 0.15).</summary>
        public double LearningRateInput { get; private set; }

        public int InputCount
        {
            get { return ImageSide * ImageSide; }
        }

        public int OutputCount
        {
            get { return ClassNames.Count; }
        }

        private ClassifierModel(int imageSide, int hiddenCount, IList<string> classNames)
        {
            ImageSide = imageSide;
            HiddenCount = hiddenCount;
            ClassNames = new List<string>(classNames);
            net = new NeuralNet(InputCount, HiddenCount, OutputCount);
            LearningRateOutput = 0.2;
            LearningRateInput = 0.15;
        }

        /// <summary>Changes both learning rates (needs the adjustable BackNeural 1.cs).</summary>
        public void SetLearningRates(double outputRate, double inputRate)
        {
            if (outputRate <= 0.0 || inputRate <= 0.0)
                throw new ArgumentException("Learning rates must be greater than zero.");
            LearningRateOutput = outputRate;
            LearningRateInput = inputRate;
            net.setLearningRates(outputRate, inputRate);
        }

        // ------------------------------------------------------------------
        // Creation / initialisation
        // ------------------------------------------------------------------

        /// <summary>
        /// Builds a network and initialises its weights with a Xavier/Glorot-style range:
        /// uniform in [-m*sqrt(6/(fanIn+fanOut)), +m*sqrt(6/(fanIn+fanOut))], m = initMultiplier.
        /// (With ~1000+ inputs the default +-0.5 range would saturate every hidden neuron immediately.)
        /// </summary>
        public static ClassifierModel Create(int imageSide, int hiddenCount, IList<string> classNames,
                                             double initMultiplier, double learningRateOutput,
                                             double learningRateInput, int seed)
        {
            if (classNames == null || classNames.Count < 2)
                throw new ArgumentException("At least two classes are required.");
            if (hiddenCount < 1)
                throw new ArgumentException("The hidden layer needs at least one neuron.");
            if (initMultiplier <= 0.0)
                throw new ArgumentException("The initial-weight multiplier must be greater than zero.");

            ClassifierModel model = new ClassifierModel(imageSide, hiddenCount, classNames);
            model.SetLearningRates(learningRateOutput, learningRateInput);
            model.InitialiseWeights(initMultiplier, seed);
            return model;
        }

        /// <summary>
        /// The engine keeps its weights private, so initial weights are handed over through its own
        /// loadWeights() method using a temporary file in the same format saveWeights() writes.
        /// </summary>
        private void InitialiseWeights(double multiplier, int seed)
        {
            Random rng = new Random(seed);
            double inputLimit = multiplier * Math.Sqrt(6.0 / (InputCount + HiddenCount));
            double outputLimit = multiplier * Math.Sqrt(6.0 / (HiddenCount + OutputCount));
            string temp = Path.GetTempFileName();

            try
            {
                WithInvariantCulture(delegate
                {
                    using (StreamWriter sw = new StreamWriter(temp, false, Encoding.ASCII, 1 << 16))
                    {
                        long inputWeights = (long)InputCount * HiddenCount;
                        for (long i = 0; i < inputWeights; i++)
                            sw.WriteLine(Format((rng.NextDouble() * 2.0 - 1.0) * inputLimit));

                        long hiddenWeights = (long)HiddenCount * OutputCount;
                        for (long i = 0; i < hiddenWeights; i++)
                            sw.WriteLine(Format((rng.NextDouble() * 2.0 - 1.0) * outputLimit));

                        for (int i = 0; i < HiddenCount; i++) sw.WriteLine(Format(0.01)); // hidden biases
                        for (int i = 0; i < OutputCount; i++) sw.WriteLine(Format(0.01)); // output biases
                    }
                    net.loadWeights(temp);
                });
            }
            finally
            {
                try { File.Delete(temp); } catch (IOException) { }
            }
        }

        // ------------------------------------------------------------------
        // Prediction / evaluation
        // ------------------------------------------------------------------

        /// <summary>Forward pass. Returns the activation (0..1) of every output neuron.</summary>
        public double[] Predict(double[] pixels)
        {
            if (pixels == null || pixels.Length != InputCount)
                throw new ArgumentException("Expected " + InputCount + " input values.");

            for (int i = 0; i < pixels.Length; i++) net.setInputs(i, pixels[i]);
            net.run();

            double[] outputs = new double[OutputCount];
            for (int k = 0; k < outputs.Length; k++) outputs[k] = net.getOuputData(k);
            return outputs;
        }

        public static int ArgMax(double[] values)
        {
            int best = 0;
            for (int i = 1; i < values.Length; i++)
                if (values[i] > values[best]) best = i;
            return best;
        }

        /// <summary>Computes accuracy, MSE, confusion matrix, precision/recall/F1. Null if no samples.</summary>
        public ClassificationMetrics Evaluate(IList<ImageSample> samples)
        {
            if (samples == null || samples.Count == 0) return null;

            List<int> actual = new List<int>(samples.Count);
            List<int> predicted = new List<int>(samples.Count);
            double sumSquares = 0.0;

            foreach (ImageSample s in samples)
            {
                double[] outputs = Predict(s.Pixels);
                for (int k = 0; k < outputs.Length; k++)
                {
                    double target = (k == s.ClassIndex) ? 1.0 : 0.0;
                    double diff = target - outputs[k];
                    sumSquares += diff * diff;
                }
                actual.Add(s.ClassIndex);
                predicted.Add(ArgMax(outputs));
            }

            double mse = sumSquares / ((double)samples.Count * OutputCount);
            return ClassificationMetrics.Create(OutputCount, actual, predicted, mse);
        }

        // ------------------------------------------------------------------
        // Training
        // ------------------------------------------------------------------

        /// <summary>
        /// One pass over the training set in random order (online back-propagation).
        /// Returns false if cancelled before the epoch finished.
        /// loss = running mean squared error; accuracy = running accuracy during the epoch.
        /// </summary>
        public bool TrainOneEpoch(IList<ImageSample> train, Random rng, CancellationToken token,
                                  out double loss, out double accuracy)
        {
            int[] order = Enumerable.Range(0, train.Count).ToArray();
            ImageDataset.Shuffle(order, rng);

            double sumSquares = 0.0;
            int correct = 0;

            foreach (int index in order)
            {
                if (token.IsCancellationRequested)
                {
                    loss = double.NaN;
                    accuracy = double.NaN;
                    return false;
                }

                ImageSample s = train[index];
                for (int i = 0; i < s.Pixels.Length; i++) net.setInputs(i, s.Pixels[i]);
                for (int k = 0; k < OutputCount; k++) net.setDesiredOutput(k, k == s.ClassIndex ? 1.0 : 0.0);

                // learn() = forward pass + back-propagation + weight/bias update.
                net.learn();

                // The output activations still hold the forward pass computed inside learn().
                int best = 0;
                double bestValue = double.NegativeInfinity;
                for (int k = 0; k < OutputCount; k++)
                {
                    double o = net.getOuputData(k);
                    double t = (k == s.ClassIndex) ? 1.0 : 0.0;
                    sumSquares += (t - o) * (t - o);
                    if (o > bestValue)
                    {
                        bestValue = o;
                        best = k;
                    }
                }
                if (best == s.ClassIndex) correct++;
            }

            loss = sumSquares / ((double)train.Count * OutputCount);
            accuracy = (double)correct / train.Count;
            return true;
        }

        /// <summary>
        /// Trains for up to <paramref name="epochs"/> epochs. After each epoch the test set (if any)
        /// is evaluated and <paramref name="onEpoch"/> is called (from the calling thread).
        /// Stops early (without error) when the token is cancelled.
        /// </summary>
        public List<EpochResult> Train(IList<ImageSample> train, IList<ImageSample> test, int epochs, int seed,
                                       Action<EpochResult> onEpoch, CancellationToken token)
        {
            if (train == null || train.Count == 0)
                throw new InvalidOperationException("There are no training images.");

            List<EpochResult> history = new List<EpochResult>();
            Random rng = new Random(seed);
            Stopwatch clock = Stopwatch.StartNew();

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                double loss;
                double accuracy;
                if (!TrainOneEpoch(train, rng, token, out loss, out accuracy)) break;

                if (double.IsNaN(loss) || double.IsInfinity(loss))
                    throw new InvalidOperationException(
                        "Training diverged (the loss is not a number). Create the network again with a smaller initial-weight multiplier.");

                EpochResult result = new EpochResult();
                result.Epoch = epoch;
                result.TrainLoss = loss;
                result.TrainAccuracy = accuracy;
                result.ElapsedSeconds = clock.Elapsed.TotalSeconds;

                if (test != null && test.Count > 0)
                {
                    ClassificationMetrics testMetrics = Evaluate(test);
                    result.TestLoss = testMetrics.Mse;
                    result.TestAccuracy = testMetrics.Accuracy;
                }

                history.Add(result);
                if (onEpoch != null) onEpoch(result);
            }
            return history;
        }

        // ------------------------------------------------------------------
        // Saving / loading
        // ------------------------------------------------------------------

        /// <summary>
        /// Writes the weights with the engine's own saveWeights() and a small ".meta" text file next to it
        /// (image size, hidden neurons, class names) so the model can be rebuilt later.
        /// </summary>
        public void Save(string weightsPath)
        {
            WithInvariantCulture(delegate { net.saveWeights(weightsPath); });

            string[] meta =
            {
                "BackPropNNImages model v1",
                "imageSide=" + ImageSide.ToString(CultureInfo.InvariantCulture),
                "hidden=" + HiddenCount.ToString(CultureInfo.InvariantCulture),
                "classes=" + string.Join("|", ClassNames.Select(c => c.Replace('|', '_'))),
                "learningRateOutput=" + LearningRateOutput.ToString("R", CultureInfo.InvariantCulture),
                "learningRateInput=" + LearningRateInput.ToString("R", CultureInfo.InvariantCulture)
            };
            File.WriteAllLines(MetaPath(weightsPath), meta, Encoding.UTF8);
        }

        public static string MetaPath(string weightsPath)
        {
            return weightsPath + ".meta";
        }

        public static ClassifierModel Load(string weightsPath)
        {
            string metaPath = MetaPath(weightsPath);
            if (!File.Exists(weightsPath))
                throw new FileNotFoundException("Weights file not found.", weightsPath);
            if (!File.Exists(metaPath))
                throw new FileNotFoundException(
                    "The description file that belongs to these weights is missing. Keep '" +
                    Path.GetFileName(metaPath) + "' in the same folder as the weights file.", metaPath);

            Dictionary<string, string> values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (string line in File.ReadAllLines(metaPath, Encoding.UTF8))
            {
                int eq = line.IndexOf('=');
                if (eq > 0) values[line.Substring(0, eq).Trim()] = line.Substring(eq + 1).Trim();
            }

            string sideText = null, hiddenText = null, classText = null;
            int side = 0, hidden = 0;
            if (!values.TryGetValue("imageSide", out sideText) || !int.TryParse(sideText, NumberStyles.Integer, CultureInfo.InvariantCulture, out side) ||
                !values.TryGetValue("hidden", out hiddenText) || !int.TryParse(hiddenText, NumberStyles.Integer, CultureInfo.InvariantCulture, out hidden) ||
                !values.TryGetValue("classes", out classText))
            {
                throw new InvalidDataException("The model description file is damaged.");
            }

            List<string> classes = classText.Split(new[] { '|' }, StringSplitOptions.RemoveEmptyEntries).ToList();
            if (classes.Count < 2 || side < 4 || hidden < 1)
                throw new InvalidDataException("The model description file contains invalid values.");

            // Convert.ToDouble(null) silently returns 0, so verify the weight count first.
            long expected = (long)side * side * hidden + (long)hidden * classes.Count + hidden + classes.Count;
            long actual = File.ReadLines(weightsPath).LongCount();
            if (actual != expected)
                throw new InvalidDataException("The weights file has " + actual + " values but the model needs " + expected + ".");

            ClassifierModel model = new ClassifierModel(side, hidden, classes);

            // Learning rates are optional in the description file (older files do not have them).
            double lrOut = 0.2, lrIn = 0.15, parsed;
            string lrText;
            if (values.TryGetValue("learningRateOutput", out lrText) &&
                double.TryParse(lrText, NumberStyles.Float, CultureInfo.InvariantCulture, out parsed) && parsed > 0.0)
                lrOut = parsed;
            if (values.TryGetValue("learningRateInput", out lrText) &&
                double.TryParse(lrText, NumberStyles.Float, CultureInfo.InvariantCulture, out parsed) && parsed > 0.0)
                lrIn = parsed;
            model.SetLearningRates(lrOut, lrIn);

            WithInvariantCulture(delegate { model.net.loadWeights(weightsPath); });
            return model;
        }

        /// <summary>True if this network was built for the same image size and the same ordered classes.</summary>
        public bool IsCompatibleWith(ImageDataset dataset)
        {
            if (dataset == null) return false;
            if (dataset.ImageSide != ImageSide) return false;
            if (dataset.ClassNames.Count != ClassNames.Count) return false;
            for (int i = 0; i < ClassNames.Count; i++)
                if (!string.Equals(dataset.ClassNames[i], ClassNames[i], StringComparison.OrdinalIgnoreCase))
                    return false;
            return true;
        }

        // ------------------------------------------------------------------
        // Helpers
        // ------------------------------------------------------------------

        private static string Format(double value)
        {
            return value.ToString("R", CultureInfo.InvariantCulture);
        }

        /// <summary>
        /// The engine reads / writes numbers with the current culture. Forcing the invariant culture
        /// makes weight files portable (always "0.25", never "0,25").
        /// </summary>
        private static void WithInvariantCulture(Action action)
        {
            CultureInfo previous = Thread.CurrentThread.CurrentCulture;
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            try
            {
                action();
            }
            finally
            {
                Thread.CurrentThread.CurrentCulture = previous;
            }
        }
    }
}
