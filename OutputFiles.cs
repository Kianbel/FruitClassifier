using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Windows.Forms;

namespace BackPropNNImages
{
    /// <summary>Helpers for the CSV files the app produces (training log, experiment results).</summary>
    public static class OutputFiles
    {
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        /// <summary>"outputs" folder next to the executable (created on demand).</summary>
        public static string GetOutputFolder()
        {
            string dir = Path.Combine(Application.StartupPath, "outputs");
            Directory.CreateDirectory(dir);
            return dir;
        }

        public static string NewFilePath(string prefix, string extension)
        {
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", Inv);
            return Path.Combine(GetOutputFolder(), prefix + "_" + stamp + extension);
        }

        private static string Num(double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value)) return "";
            return value.ToString("0.000000", Inv);
        }

        public static void WriteExperimentCsv(string path, IEnumerable<ExperimentResult> results)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("run,hidden_neurons,init_multiplier,learning_rate_output,learning_rate_input,epochs,train_loss,train_accuracy,test_loss,test_accuracy,seconds");
            foreach (ExperimentResult r in results)
            {
                sb.AppendLine(string.Join(",", new[]
                {
                    r.Run.ToString(Inv),
                    r.Hidden.ToString(Inv),
                    r.InitMultiplier.ToString("0.###", Inv),
                    r.LearningRateOutput.ToString("0.####", Inv),
                    r.LearningRateInput.ToString("0.####", Inv),
                    r.Epochs.ToString(Inv),
                    Num(r.TrainLoss),
                    Num(r.TrainAccuracy),
                    Num(r.TestLoss),
                    Num(r.TestAccuracy),
                    r.Seconds.ToString("0.0", Inv)
                }));
            }
            File.WriteAllText(path, sb.ToString(), Encoding.UTF8);
        }

        /// <summary>
        /// Writes one CSV line per epoch while training runs (flushed immediately,
        /// so the file is complete even if the program is closed during training).
        /// </summary>
        public class TrainingLogWriter : IDisposable
        {
            private readonly StreamWriter writer;
            public string FilePath { get; private set; }

            public TrainingLogWriter(string path)
            {
                FilePath = path;
                writer = new StreamWriter(path, false, Encoding.UTF8);
                writer.AutoFlush = true;
                writer.WriteLine("epoch,train_loss,train_accuracy,test_loss,test_accuracy,elapsed_seconds");
            }

            public void Append(EpochResult r)
            {
                writer.WriteLine(string.Join(",", new[]
                {
                    r.Epoch.ToString(Inv),
                    Num(r.TrainLoss),
                    Num(r.TrainAccuracy),
                    Num(r.TestLoss),
                    Num(r.TestAccuracy),
                    r.ElapsedSeconds.ToString("0.0", Inv)
                }));
            }

            public void Dispose()
            {
                writer.Dispose();
            }
        }
    }
}
