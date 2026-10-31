using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace BackPropNNImages
{
    /// <summary>
    /// Evaluation results for a multi-class classifier:
    /// accuracy, MSE loss, confusion matrix and per-class precision / recall / F1.
    /// </summary>
    public class ClassificationMetrics
    {
        public int SampleCount;

        /// <summary>Fraction of correctly classified samples (0..1).</summary>
        public double Accuracy;

        /// <summary>Mean squared error between network outputs and one-hot targets.</summary>
        public double Mse;

        /// <summary>Confusion[actual, predicted].</summary>
        public int[,] Confusion;

        public int[] Support;
        public double[] Precision;
        public double[] Recall;
        public double[] F1;

        public double MacroPrecision;
        public double MacroRecall;
        public double MacroF1;

        public static ClassificationMetrics Create(int classCount, IList<int> actual, IList<int> predicted, double mse)
        {
            ClassificationMetrics m = new ClassificationMetrics();
            m.SampleCount = actual.Count;
            m.Mse = mse;
            m.Confusion = new int[classCount, classCount];
            m.Support = new int[classCount];
            m.Precision = new double[classCount];
            m.Recall = new double[classCount];
            m.F1 = new double[classCount];

            int correct = 0;
            for (int i = 0; i < actual.Count; i++)
            {
                m.Confusion[actual[i], predicted[i]]++;
                m.Support[actual[i]]++;
                if (actual[i] == predicted[i]) correct++;
            }
            m.Accuracy = actual.Count == 0 ? 0.0 : (double)correct / actual.Count;

            for (int c = 0; c < classCount; c++)
            {
                int truePositive = m.Confusion[c, c];
                int predictedAsC = 0;
                int actualC = 0;
                for (int k = 0; k < classCount; k++)
                {
                    predictedAsC += m.Confusion[k, c];
                    actualC += m.Confusion[c, k];
                }

                // A class that is never predicted (or never present) scores 0 instead of dividing by zero.
                m.Precision[c] = predictedAsC == 0 ? 0.0 : (double)truePositive / predictedAsC;
                m.Recall[c] = actualC == 0 ? 0.0 : (double)truePositive / actualC;
                double pr = m.Precision[c] + m.Recall[c];
                m.F1[c] = pr == 0.0 ? 0.0 : 2.0 * m.Precision[c] * m.Recall[c] / pr;

                m.MacroPrecision += m.Precision[c] / classCount;
                m.MacroRecall += m.Recall[c] / classCount;
                m.MacroF1 += m.F1[c] / classCount;
            }
            return m;
        }

        /// <summary>Plain-text report (used by the Export Report button).</summary>
        public string ToText(IList<string> classNames, string title)
        {
            CultureInfo inv = CultureInfo.InvariantCulture;
            StringBuilder sb = new StringBuilder();
            sb.AppendLine(title);
            sb.AppendLine(new string('-', title.Length));
            sb.AppendLine("Samples   : " + SampleCount);
            sb.AppendLine("Accuracy  : " + (Accuracy * 100.0).ToString("0.00", inv) + " %");
            sb.AppendLine("MSE loss  : " + Mse.ToString("0.000000", inv));
            sb.AppendLine("Macro precision / recall / F1 : "
                + MacroPrecision.ToString("0.000", inv) + " / "
                + MacroRecall.ToString("0.000", inv) + " / "
                + MacroF1.ToString("0.000", inv));
            sb.AppendLine();
            sb.AppendLine("Per-class results");
            sb.AppendLine(Pad("Class", 24) + Pad("Support", 10) + Pad("Precision", 11) + Pad("Recall", 9) + "F1");
            for (int c = 0; c < classNames.Count; c++)
            {
                sb.AppendLine(Pad(classNames[c], 24) + Pad(Support[c].ToString(inv), 10)
                    + Pad(Precision[c].ToString("0.000", inv), 11)
                    + Pad(Recall[c].ToString("0.000", inv), 9)
                    + F1[c].ToString("0.000", inv));
            }
            sb.AppendLine();
            sb.AppendLine("Confusion matrix (rows = actual, columns = predicted)");
            sb.Append(Pad("", 24));
            for (int c = 0; c < classNames.Count; c++) sb.Append(Pad(Shorten(classNames[c], 9), 10));
            sb.AppendLine();
            for (int r = 0; r < classNames.Count; r++)
            {
                sb.Append(Pad(classNames[r], 24));
                for (int c = 0; c < classNames.Count; c++) sb.Append(Pad(Confusion[r, c].ToString(inv), 10));
                sb.AppendLine();
            }
            return sb.ToString();
        }

        private static string Pad(string text, int width)
        {
            if (text.Length >= width) return text.Substring(0, width - 1) + " ";
            return text.PadRight(width);
        }

        private static string Shorten(string text, int max)
        {
            return text.Length <= max ? text : text.Substring(0, max);
        }
    }
}
