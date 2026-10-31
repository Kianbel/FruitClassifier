using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.Windows.Forms;

namespace BackPropNNImages
{
    /// <summary>
    /// Small self-drawn line chart (no extra assemblies needed). Used for the loss and accuracy curves.
    /// </summary>
    public class LineChart : Control
    {
        private readonly List<List<double>> values = new List<List<double>>();
        private readonly List<string> names = new List<string>();
        private readonly List<Color> colors = new List<Color>();

        public string Title { get; set; }

        /// <summary>When true the y axis is fixed to 0..1 (accuracy); otherwise it follows the data.</summary>
        public bool FixedUnitRange { get; set; }

        /// <summary>Format of the y axis labels.</summary>
        public string YFormat { get; set; }

        public LineChart()
        {
            DoubleBuffered = true;
            ResizeRedraw = true;
            BackColor = Color.White;
            Font = new Font("Segoe UI", 9f);
            YFormat = "0.###";
            Title = "";
        }

        public int AddSeries(string name, Color color)
        {
            values.Add(new List<double>());
            names.Add(name);
            colors.Add(color);
            return values.Count - 1;
        }

        public void AddPoint(int series, double value)
        {
            values[series].Add(value);
            Invalidate();
        }

        public void ClearData()
        {
            foreach (List<double> v in values) v.Clear();
            Invalidate();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;
            g.Clear(BackColor);

            Rectangle plot = new Rectangle(58, 40, Math.Max(20, Width - 58 - 20), Math.Max(20, Height - 40 - 34));

            using (Font titleFont = new Font("Segoe UI Semibold", 10.5f))
            using (Brush textBrush = new SolidBrush(Color.FromArgb(70, 80, 95)))
            using (Brush titleBrush = new SolidBrush(Color.FromArgb(25, 40, 60)))
            using (Pen gridPen = new Pen(Color.FromArgb(232, 236, 242)))
            using (Pen axisPen = new Pen(Color.FromArgb(170, 178, 190)))
            {
                g.DrawString(Title, titleFont, titleBrush, 12, 10);

                // y range
                double yMax = 1.0;
                if (!FixedUnitRange)
                {
                    yMax = 0.0;
                    foreach (List<double> series in values)
                        foreach (double v in series)
                            if (!double.IsNaN(v) && !double.IsInfinity(v) && v > yMax) yMax = v;
                    if (yMax <= 0.0) yMax = 1.0;
                    yMax *= 1.1;
                }

                // grid + y labels
                StringFormat right = new StringFormat();
                right.Alignment = StringAlignment.Far;
                for (int i = 0; i <= 4; i++)
                {
                    float y = plot.Bottom - i * plot.Height / 4f;
                    g.DrawLine(i == 0 ? axisPen : gridPen, plot.Left, y, plot.Right, y);
                    string label = (yMax * i / 4.0).ToString(YFormat, CultureInfo.InvariantCulture);
                    g.DrawString(label, Font, textBrush, new RectangleF(0, y - 9, plot.Left - 6, 18), right);
                }
                right.Dispose();
                g.DrawLine(axisPen, plot.Left, plot.Top, plot.Left, plot.Bottom);

                // x range
                int n = 0;
                foreach (List<double> series in values) n = Math.Max(n, series.Count);

                if (n == 0)
                {
                    using (StringFormat center = new StringFormat())
                    {
                        center.Alignment = StringAlignment.Center;
                        center.LineAlignment = StringAlignment.Center;
                        g.DrawString("No data yet", Font, textBrush, plot, center);
                    }
                }
                else
                {
                    g.DrawString("epoch 1", Font, textBrush, plot.Left - 10, plot.Bottom + 6);
                    using (StringFormat far = new StringFormat())
                    {
                        far.Alignment = StringAlignment.Far;
                        g.DrawString("epoch " + n, Font, textBrush,
                            new RectangleF(plot.Right - 100, plot.Bottom + 6, 110, 18), far);
                    }
                }

                // series
                for (int s = 0; s < values.Count; s++)
                {
                    List<PointF> points = new List<PointF>();
                    for (int k = 0; k < values[s].Count; k++)
                    {
                        double v = values[s][k];
                        if (double.IsNaN(v) || double.IsInfinity(v)) continue;
                        float x = n <= 1 ? plot.Left : plot.Left + k * (float)plot.Width / (n - 1);
                        double clamped = Math.Min(Math.Max(v, 0.0), yMax);
                        float y = plot.Bottom - (float)(clamped / yMax) * plot.Height;
                        points.Add(new PointF(x, y));
                    }

                    using (Pen pen = new Pen(colors[s], 2.2f))
                    using (Brush dot = new SolidBrush(colors[s]))
                    {
                        pen.LineJoin = LineJoin.Round;
                        if (points.Count >= 2) g.DrawLines(pen, points.ToArray());
                        if (points.Count >= 1)
                        {
                            PointF last = points[points.Count - 1];
                            g.FillEllipse(dot, last.X - 3.5f, last.Y - 3.5f, 7f, 7f);
                        }
                    }
                }

                // legend (top right)
                float lx = Width - 16;
                for (int s = values.Count - 1; s >= 0; s--)
                {
                    SizeF size = g.MeasureString(names[s], Font);
                    lx -= size.Width;
                    g.DrawString(names[s], Font, textBrush, lx, 11);
                    using (Brush b = new SolidBrush(colors[s]))
                        g.FillRectangle(b, lx - 16, 17, 11, 11);
                    lx -= 28;
                }
            }
        }
    }
}
