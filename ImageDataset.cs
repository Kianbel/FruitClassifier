using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

namespace BackPropNNImages
{
    /// <summary>
    /// One image converted to a normalised feature vector (values in [0,1]).
    /// </summary>
    public class ImageSample
    {
        public string FilePath;
        public string ClassName;
        public int ClassIndex;
        public double[] Pixels;
    }

    /// <summary>
    /// Data pipeline: parses file names into class labels, loads and resizes images,
    /// MinMax-scales every pixel to [0,1] (grey / 255) and splits into train / test sets.
    ///
    /// Recommended file names:  "[fruit_name] (number).jpeg|.jpg|.png"
    /// e.g.  "apple (12).png"  ->  class "apple"
    /// Other common styles are accepted too (see ExtractClassName).
    /// </summary>
    public class ImageDataset
    {
        public static readonly string[] SupportedExtensions = { ".jpg", ".jpeg", ".png" };

        // Trailing "(12)", "[3]", "{7}" ... (also "(copy)").
        private static readonly Regex TrailingBracket =
            new Regex(@"\s*[\(\[\{][^\)\]\}]*[\)\]\}]\s*$", RegexOptions.Compiled);

        // Trailing " - Copy", "_copy", " copy 2" (added by Windows or by hand).
        private static readonly Regex TrailingCopy =
            new Regex(@"[\s_\-]+copy(?:\s*\d+)?\s*$", RegexOptions.Compiled | RegexOptions.IgnoreCase);

        // Trailing separator + number when there are no brackets: "name_12", "name-12", "name 12", "name12".
        private static readonly Regex TrailingNumber =
            new Regex(@"[\s_\-\.]*\d+\s*$", RegexOptions.Compiled);

        /// <summary>Width = height of every image after resizing (network input = side * side).</summary>
        public int ImageSide { get; private set; }

        public List<ImageSample> Samples { get; private set; }

        /// <summary>Class names, sorted alphabetically. Index in this list = output neuron index.</summary>
        public List<string> ClassNames { get; private set; }

        /// <summary>Files that could not be used (corrupt, unreadable...), with the reason.</summary>
        public List<string> SkippedFiles { get; private set; }

        private ImageDataset()
        {
            Samples = new List<ImageSample>();
            ClassNames = new List<string>();
            SkippedFiles = new List<string>();
        }

        public int InputCount
        {
            get { return ImageSide * ImageSide; }
        }

        // ------------------------------------------------------------------
        // Label parsing
        // ------------------------------------------------------------------

        public static bool IsSupported(string path)
        {
            string ext = Path.GetExtension(path);
            return SupportedExtensions.Contains(ext, StringComparer.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Extracts the fruit name from a file name. Accepted styles (extension ignored, case ignored):
        ///   "apple (3).jpg"   "apple(3).jpg"   "apple [3].png"   "apple_3.jpg"   "apple-3.jpg"
        ///   "apple 3.jpg"   "apple (3) - Copy.jpg"   "apple (3) (1).jpg"   "Apple (12).png"
        /// Underscores count as spaces, so "dragon_fruit (1)" and "dragon_fruit(2)" are one class.
        /// If nothing is left (for example "12.png" or "(3).png") the name of the parent folder is used.
        /// Note: a class name that itself ends in a number (such as "h1n1") must be written with brackets,
        /// e.g. "h1n1 (4).png", otherwise the trailing number is treated as the image number.
        /// </summary>
        public static string ExtractClassName(string path)
        {
            string stem = Path.GetFileNameWithoutExtension(path) ?? "";
            string name = stem.Trim();
            bool hadBracket = false;
            string previous;

            // Peel off trailing "(1)", "[2]", "- Copy" ... in any order until nothing changes.
            do
            {
                previous = name;
                name = TrailingCopy.Replace(name, "");
                Match m = TrailingBracket.Match(name);
                if (m.Success)
                {
                    name = name.Substring(0, m.Index);
                    hadBracket = true;
                }
                name = name.TrimEnd(' ', '_', '-', '.');
            }
            while (name != previous);

            if (!hadBracket) name = TrailingNumber.Replace(name, "");

            name = Regex.Replace(name.Replace('_', ' '), @"\s+", " ").Trim();

            if (name.Length == 0 || Regex.IsMatch(name, @"^\d+$"))
            {
                string folder = Path.GetFileName(Path.GetDirectoryName(path) ?? "");
                name = string.IsNullOrEmpty(folder) ? stem.Trim() : folder;
            }
            return name;
        }

        // ------------------------------------------------------------------
        // Loading
        // ------------------------------------------------------------------

        /// <summary>
        /// Loads every supported image in <paramref name="files"/>. Unreadable files are skipped
        /// (and listed in SkippedFiles) instead of crashing the whole load.
        /// </summary>
        public static ImageDataset Load(IEnumerable<string> files, int side, Action<int, int> progress)
        {
            if (side < 4) throw new ArgumentOutOfRangeException("side");

            ImageDataset ds = new ImageDataset();
            ds.ImageSide = side;

            List<string> valid = files
                .Where(IsSupported)
                .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
                .ToList();

            // key (lower-case) -> display name (first spelling seen)
            Dictionary<string, string> display = new Dictionary<string, string>();
            List<ImageSample> loaded = new List<ImageSample>();

            for (int i = 0; i < valid.Count; i++)
            {
                string file = valid[i];
                try
                {
                    double[] pixels = LoadPixels(file, side);
                    string name = ExtractClassName(file);
                    string key = name.ToLowerInvariant();
                    if (!display.ContainsKey(key)) display[key] = name;

                    ImageSample s = new ImageSample();
                    s.FilePath = file;
                    s.ClassName = key; // temporary: replaced by display name below
                    s.Pixels = pixels;
                    loaded.Add(s);
                }
                catch (Exception ex)
                {
                    ds.SkippedFiles.Add(Path.GetFileName(file) + " - " + ex.Message);
                }

                if (progress != null) progress(i + 1, valid.Count);
            }

            List<string> keys = display.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();
            foreach (string key in keys) ds.ClassNames.Add(display[key]);

            foreach (ImageSample s in loaded)
            {
                int idx = keys.IndexOf(s.ClassName);
                s.ClassIndex = idx;
                s.ClassName = ds.ClassNames[idx];
                ds.Samples.Add(s);
            }

            return ds;
        }

        /// <summary>
        /// Reads an image file, converts to grey, resizes to side x side and scales to [0,1].
        /// Pixel order is row by row (index = y * side + x).
        /// </summary>
        public static double[] LoadPixels(string path, int side)
        {
            using (Bitmap source = new Bitmap(path))
            {
                return BitmapToPixels(source, side);
            }
        }

        public static double[] BitmapToPixels(Bitmap source, int side)
        {
            using (Bitmap resized = new Bitmap(side, side, PixelFormat.Format24bppRgb))
            {
                using (Graphics g = Graphics.FromImage(resized))
                using (ImageAttributes attributes = new ImageAttributes())
                {
                    attributes.SetWrapMode(WrapMode.TileFlipXY);
                    g.Clear(Color.White);
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                    g.CompositingQuality = CompositingQuality.HighQuality;
                    g.DrawImage(source,
                        new Rectangle(0, 0, side, side),
                        0, 0, source.Width, source.Height,
                        GraphicsUnit.Pixel, attributes);
                }

                BitmapData data = resized.LockBits(new Rectangle(0, 0, side, side),
                    ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                try
                {
                    byte[] buffer = new byte[Math.Abs(data.Stride) * side];
                    Marshal.Copy(data.Scan0, buffer, 0, buffer.Length);

                    double[] pixels = new double[side * side];
                    for (int y = 0; y < side; y++)
                    {
                        for (int x = 0; x < side; x++)
                        {
                            int o = y * data.Stride + x * 3;
                            double b = buffer[o];
                            double gr = buffer[o + 1];
                            double r = buffer[o + 2];
                            // Images are already grey, but the luminance formula keeps colour files safe too.
                            pixels[y * side + x] = (0.299 * r + 0.587 * gr + 0.114 * b) / 255.0;
                        }
                    }
                    return pixels;
                }
                finally
                {
                    resized.UnlockBits(data);
                }
            }
        }

        /// <summary>Turns a feature vector back into a small bitmap (used for previews).</summary>
        public static Bitmap PixelsToBitmap(double[] pixels, int side)
        {
            Bitmap bmp = new Bitmap(side, side);
            for (int y = 0; y < side; y++)
            {
                for (int x = 0; x < side; x++)
                {
                    int v = (int)Math.Round(Math.Max(0.0, Math.Min(1.0, pixels[y * side + x])) * 255.0);
                    bmp.SetPixel(x, y, Color.FromArgb(v, v, v));
                }
            }
            return bmp;
        }

        // ------------------------------------------------------------------
        // Splitting
        // ------------------------------------------------------------------

        /// <summary>
        /// Stratified random split: every class keeps (about) the same train/test ratio.
        /// A class with 2+ images always keeps at least one image in each set;
        /// a class with a single image goes entirely to training.
        /// </summary>
        public void Split(double testFraction, int seed, out List<ImageSample> train, out List<ImageSample> test)
        {
            Random rng = new Random(seed);
            train = new List<ImageSample>();
            test = new List<ImageSample>();

            foreach (IGrouping<int, ImageSample> group in Samples.GroupBy(s => s.ClassIndex).OrderBy(g => g.Key))
            {
                List<ImageSample> items = group.ToList();
                Shuffle(items, rng);

                int testCount = 0;
                if (testFraction > 0.0 && items.Count >= 2)
                {
                    testCount = (int)Math.Round(items.Count * testFraction);
                    testCount = Math.Max(1, Math.Min(testCount, items.Count - 1));
                }

                for (int i = 0; i < items.Count; i++)
                {
                    if (i < testCount) test.Add(items[i]);
                    else train.Add(items[i]);
                }
            }

            // Mix classes so that training order is not grouped by class.
            Shuffle(train, rng);
            Shuffle(test, rng);
        }

        /// <summary>Number of samples per class index.</summary>
        public int[] CountPerClass(IEnumerable<ImageSample> samples)
        {
            int[] counts = new int[ClassNames.Count];
            foreach (ImageSample s in samples) counts[s.ClassIndex]++;
            return counts;
        }

        /// <summary>Fisher-Yates shuffle.</summary>
        public static void Shuffle<T>(IList<T> list, Random rng)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                T tmp = list[i];
                list[i] = list[j];
                list[j] = tmp;
            }
        }
    }
}
