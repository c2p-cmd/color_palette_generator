# Image Color Palette Generator

A powerful and interactive tool for extracting dominant colors from images using K-means clustering. Built with Marimo for an intuitive web-based interface, this application allows you to upload images and extract the most prominent colors with customizable parameters.

## 🌈 Features

- **Interactive Web Interface**: Upload images via drag-and-drop or file selection
- **Customizable Color Extraction**: Choose between 1-6 dominant colors to extract
- **Adjustable Precision**: Control the clustering iterations (100-10,000) for better accuracy
- **Multiple Output Formats**: View results as color palettes, pie charts, and individual color cards
- **Color Format Support**: Get RGB values and hex codes for each extracted color
- **Image Format Support**: Works with PNG, JPG, and JPEG files
- **Auto Image Resizing**: Automatically resizes large images (>1000px) while maintaining aspect ratio

## 🛠 Technology Stack

- **[Marimo](https://marimo.io/)**: Interactive Python notebooks for the web interface
- **NumPy**: Efficient numerical computations for the K-means algorithm
- **Matplotlib**: Visualization and plotting of results
- **Pillow (PIL)**: Image processing and format handling
- **Custom K-means Implementation**: Pure NumPy implementation for color clustering

## 📋 Requirements

- Python 3.11 or higher
- Dependencies (automatically managed via script metadata):
  - `matplotlib==3.10.6`
  - `numpy==2.3.3` 
  - `pillow==11.3.0`
  - `marimo` (for the interactive interface)

## 🚀 Installation & Usage

### Method 1: Using Marimo (Recommended)

1. **Install Marimo**:
   ```bash
   pip install marimo
   ```

2. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd average_color_extraction
   ```

3. **Run the application**:
   ```bash
   marimo run color_pallete.py
   ```

4. **Open your browser** and navigate to the displayed URL (typically `http://localhost:2718`)

### Method 2: Direct Python Execution

The script includes dependency metadata that makes it runnable directly:

```bash
uv run --with marimo edit --sandbox color_pallete.py
# or 
uv run --with marimo run --sandbox color_pallete.py
```

## 🎨 How It Works

### The Algorithm

The application uses a custom implementation of the K-means clustering algorithm:

1. **Image Processing**: Uploaded images are converted to numpy arrays and resized if necessary
2. **Color Space Transformation**: RGB pixels are reshaped into a 2D array for clustering
3. **K-means Clustering**: The algorithm groups similar colors together:
   - Randomly initializes cluster centroids
   - Iteratively assigns pixels to nearest centroids
   - Updates centroids based on assigned pixels
   - Converges when centroids stop moving significantly
4. **Result Generation**: Extracts dominant colors and calculates their distribution

### User Interface

- **Image Upload**: Drag and drop or click to upload PNG/JPG/JPEG files
- **Color Count Slider**: Adjust the number of colors to extract (1-6)
- **Iterations Slider**: Control clustering precision (100-10,000 iterations)
- **Results Display**: 
  - Original image preview
  - Dominant color palette bar
  - Pie chart showing color distribution percentages
  - Individual color cards with RGB and hex values

## 📊 Output Formats

The application provides multiple ways to view your extracted colors:

1. **Color Palette Bar**: Visual representation showing relative dominance
2. **Pie Chart**: Percentage breakdown of each color's presence
3. **Color Cards**: Individual displays showing:
   - RGB values (e.g., R: 255, G: 128, B: 64)
   - Hex codes (e.g., #FF8040)
   - Pure color swatches

## 🔧 Configuration Options

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Number of Colors | 1-6 | 3 | How many dominant colors to extract |
| Max Iterations | 100-10,000 | 300 | Clustering precision (higher = more accurate) |

## 📁 Project Structure

```
average_color_extraction/
├── color_pallete.py          # Main application file
├── app/                      # Web application assets
│   ├── index.html           # HTML entry point
│   ├── assets/              # CSS, JS, and other web assets
│   ├── files/               # File handling assets
│   └── [various icons/favicons]
└── README.md                # This file
```

## 🎯 Use Cases

- **Web Design**: Extract color schemes from inspiration images
- **Brand Development**: Analyze color palettes from logos or marketing materials
- **Art Analysis**: Study the dominant colors in paintings or photographs
- **Fashion**: Determine color schemes from clothing or fabric images
- **Interior Design**: Extract colors from room photos for decoration planning

## 🔬 Technical Details

### Custom K-means Implementation

The application features a pure NumPy implementation of K-means clustering with several optimizations:

- **Robust Centroid Updates**: Handles empty clusters gracefully
- **Convergence Detection**: Uses `np.allclose()` for floating-point comparison
- **Broadcasting Optimization**: Efficient distance calculations using NumPy broadcasting
- **Memory Efficient**: Handles large images through automatic resizing

### Performance Considerations

- **Image Resizing**: Large images (>1000px) are automatically resized to improve performance
- **Aspect Ratio Preservation**: Resizing maintains original image proportions
- **Iteration Control**: Users can balance speed vs. accuracy with the iterations slider

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this tool!

## 📄 License

This project is open source. Please check the [LICENSE](./LICENSE)

## 🙋‍♂️ Support

If you encounter any issues or have questions about using the application, please:

1. Check that all dependencies are properly installed
2. Ensure your image files are in supported formats (PNG, JPG, JPEG)
3. Try reducing the number of iterations if the application runs slowly
4. Verify that your Python version is 3.11 or higher

---
