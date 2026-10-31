using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;

namespace BackPropNNImages
{
    /// <summary>Result of one hyper-parameter configuration evaluated at one epoch checkpoint.</summary>
    public class ExperimentResult
    {
        public int Run;
        public int Hidden;
        public double InitMultiplier;
        public double LearningRateOutput;
        public double LearningRateInput;
        public int Epochs;
        public double TrainLoss;
        public double TrainAccuracy;
        public double TestLoss;
        public double TestAccuracy;
        public double Seconds;
    }

    /// <summary>
    /// Systematic grid search over hidden-layer size, initial-weight multiplier, learning rate and epoch count.
    /// For every (hidden, multiplier, learning rate) combination ONE network is trained up to the largest
    /// epoch value, and it is evaluated (full train + test set) every time an epoch checkpoint is reached.
    /// All runs use the same seed and the same train/test split, so results are comparable.
    ///
    /// The list of learning rates is for the output layer; the input-layer rate is always
    /// (output rate x inputRateRatio), which keeps the ratio chosen in the "Create network" card.
    /// </summary>
    public static class HyperparameterExperiment
    {
        public static List<ExperimentResult> Run(
            ImageDataset dataset,
            IList<ImageSample> train,
            IList<ImageSample> test,
            IList<int> hiddenSizes,
            IList<double> initMultipliers,
            IList<double> outputLearningRates,
            double inputRateRatio,
            IList<int> epochCheckpoints,
            int seed,
            Action<ExperimentResult> onResult,
            Action<int, int> onStep,
            CancellationToken token)
        {
            List<ExperimentResult> results = new List<ExperimentResult>();
            int[] checkpoints = epochCheckpoints.Distinct().OrderBy(e => e).ToArray();
            if (checkpoints.Length == 0 || hiddenSizes.Count == 0 ||
                initMultipliers.Count == 0 || outputLearningRates.Count == 0) return results;

            int maxEpoch = checkpoints[checkpoints.Length - 1];
            int totalSteps = hiddenSizes.Count * initMultipliers.Count * outputLearningRates.Count * maxEpoch;
            int step = 0;
            int run = 0;

            foreach (int hidden in hiddenSizes)
            {
                foreach (double multiplier in initMultipliers)
                {
                    foreach (double lrOutput in outputLearningRates)
                    {
                        if (token.IsCancellationRequested) return results;

                        double lrInput = lrOutput * inputRateRatio;
                        ClassifierModel model = ClassifierModel.Create(
                            dataset.ImageSide, hidden, dataset.ClassNames, multiplier, lrOutput, lrInput, seed);
                        Random rng = new Random(seed);
                        Stopwatch clock = Stopwatch.StartNew();

                        for (int epoch = 1; epoch <= maxEpoch; epoch++)
                        {
                            double loss;
                            double accuracy;
                            if (!model.TrainOneEpoch(train, rng, token, out loss, out accuracy)) return results;

                            step++;
                            if (onStep != null) onStep(step, totalSteps);

                            if (Array.IndexOf(checkpoints, epoch) < 0) continue;

                            ClassificationMetrics trainMetrics = model.Evaluate(train);
                            ClassificationMetrics testMetrics = model.Evaluate(test);

                            run++;
                            ExperimentResult r = new ExperimentResult();
                            r.Run = run;
                            r.Hidden = hidden;
                            r.InitMultiplier = multiplier;
                            r.LearningRateOutput = lrOutput;
                            r.LearningRateInput = lrInput;
                            r.Epochs = epoch;
                            r.TrainLoss = trainMetrics.Mse;
                            r.TrainAccuracy = trainMetrics.Accuracy;
                            r.TestLoss = testMetrics.Mse;
                            r.TestAccuracy = testMetrics.Accuracy;
                            r.Seconds = clock.Elapsed.TotalSeconds;
                            results.Add(r);
                            if (onResult != null) onResult(r);
                        }
                    }
                }
            }
            return results;
        }

        /// <summary>Best = highest test accuracy, ties broken by the lowest test loss.</summary>
        public static ExperimentResult FindBest(IEnumerable<ExperimentResult> results)
        {
            return results
                .OrderByDescending(r => r.TestAccuracy)
                .ThenBy(r => r.TestLoss)
                .FirstOrDefault();
        }
    }
}
