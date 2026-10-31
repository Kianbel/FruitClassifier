using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace BackPropNNImages
{
    /// <summary>
    /// Horizontal bar chart that shows the activation (0..1) of every output neuron.
    /// The strongest class is highlighted.
    /// </summary>
    public class ProbabilityBars : Control
    {
        private string[] labels = new string[0];
        private double[] activations = new double[0];
        private int highlight = -1;

        public ProbabilityBars()
        {
            DoubleBuffered = true;
            ResizeRedraw = true;
            BackColor = Color.White;
            Font = new Font("Segoe UI", 10f);
        }

        public void SetValues(string[] classLabels, double[] classActivations)
        {
            labels = classLabels ?? new string[0];
            activations = classActivations ?? new double[0];
            highlight = -1;
            for (int i = 0; i < activations.Length; i++)
                if (highlight < 0 || activations[i] > activations[highlight]) highlight = i;
            Invalidate();
        }

        public void Clear()
        {
            labels = new string[0];
            activations = new double[0];
            highlight = -1;
            Invalidate();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;
            g.Clear(BackColor);

            using (Brush muted = new SolidBrush(Color.FromArgb(120, 130, 145)))
            using (StringFormat center = new StringFormat())
            {
                center.Alignment = StringAlignment.Center;
                center.LineAlignment = StringAlignment.Center;

                if (activations.Length == 0)
                {
                    g.DrawString("Open an image to see the network's answer", Font, muted, ClientRectangle, center);
                    return;
                }
            }

            int rows = activations.Length;
            int rowHeight = Math.Max(24, Math.Min(44, Height / rows));
            int labelWidth = Math.Min(170, Width / 3);
            int valueWidth = 64;
            int trackLeft = labelWidth + 8;
            int trackWidth = Math.Max(20, Width - trackLeft - valueWidth - 12);

            for (int i = 0; i < rows; i++)
            {
                int top = i * rowHeight;
                bool isTop = i == highlight;

                Color barColor = isTop ? Color.FromArgb(37, 99, 235) : Color.FromArgb(170, 182, 200);
                Color textColor = isTop ? Color.FromArgb(20, 35, 60) : Color.FromArgb(85, 95, 110);

                using (Font f = new Font("Segoe UI", 10f, isTop ? FontStyle.Bold : FontStyle.Regular))
                using (Brush text = new SolidBrush(textColor))
                using (Brush track = new SolidBrush(Color.FromArgb(238, 241, 246)))
                using (Brush bar = new SolidBrush(barColor))
                using (StringFormat nameFormat = new StringFormat())
                using (StringFormat valueFormat = new StringFormat())
                {
                    nameFormat.Trimming = StringTrimming.EllipsisCharacter;
                    nameFormat.FormatFlags = StringFormatFlags.NoWrap;
                    nameFormat.LineAlignment = StringAlignment.Center;
                    valueFormat.Alignment = StringAlignment.Far;
                    valueFormat.LineAlignment = StringAlignment.Center;

                    string name = i < labels.Length ? labels[i] : ("class " + i);
                    g.DrawString(name, f, text, new RectangleF(4, top, labelWidth, rowHeight), nameFormat);

                    Rectangle trackRect = new Rectangle(trackLeft, top + rowHeight / 2 - 8, trackWidth, 16);
                    FillRounded(g, track, trackRect);

                    double v = Math.Max(0.0, Math.Min(1.0, activations[i]));
                    int w = (int)Math.Round(trackRect.Width * v);
                    if (w > 0) FillRounded(g, bar, new Rectangle(trackRect.Left, trackRect.Top, Math.Max(w, 8), trackRect.Height));

                    g.DrawString((v * 100.0).ToString("0.0") + "%", f, text,
                        new RectangleF(trackRect.Right + 4, top, valueWidth + 4, rowHeight), valueFormat);
                }
            }
        }

        private static void FillRounded(Graphics g, Brush brush, Rectangle r)
        {
            int d = Math.Min(r.Height, 16);
            using (GraphicsPath path = new GraphicsPath())
            {
                path.AddArc(r.Left, r.Top, d, d, 90, 180);
                path.AddArc(r.Right - d, r.Top, d, d, 270, 180);
                path.CloseFigure();
                g.FillPath(brush, path);
            }
        }
    }
}
