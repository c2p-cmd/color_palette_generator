# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib==3.10.6",
#     "numpy==2.3.3",
#     "openai==1.109.1",
#     "pillow==11.3.0",
# ]
# ///

import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    mo.md("""# Image Color Pallete Generator""")
    return


@app.cell
def _(mo):
    img_file = mo.ui.file(
        filetypes=[".png", ".jpg", ".jpeg"],
        label="Upload Image to generate pallete",
        multiple=False,
        kind="area",
    )
    cluster_count = mo.ui.slider(
        start=1,
        stop=6,
        value=3,
        label="Number of colors to extract from image",
        show_value=True,
        full_width=True,
    )
    max_iters = mo.ui.slider(
        start=100,
        stop=10_000,
        step=100,
        value=300,
        label="Iterations to run",
        full_width=True,
        show_value=True,
    )
    mo.vstack(
        [
            img_file,
            mo.hstack([cluster_count, max_iters]),
        ],
        justify="center",
    )
    return cluster_count, img_file, max_iters


@app.cell
def _(img_file, mo):
    mo.stop(len(img_file.value) < 1)

    error = None
    if not all(ext in ['.png', '.jpg', '.jpeg'] for ext in [img_file.value[0].name.split('.')[-1]]):
        error = mo.md("## Image in invalid format")

    from PIL import Image
    from io import BytesIO

    error
    return BytesIO, Image


@app.cell
def _(np):
    class KMeans_NP:
        def __init__(self, n_clusters=3, max_iters=300, random_state=None):
            self.n_clusters = n_clusters
            self.max_iters = max_iters
            self.random_state = random_state
            self.centroids = None

        @property
        def cluster_centers_(self):
            """The coordinates of the cluster centers, shape (n_clusters, n_features)."""
            if self.centroids is None:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute 'cluster_centers_' until fit is called."
                )
            return self.centroids

        def fit(self, X):
            if self.random_state:
                np.random.seed(self.random_state)

            # 1. Initialize centroids randomly
            idx = np.random.choice(len(X), self.n_clusters, replace=False)
            self.centroids = X[idx].astype(
                X.dtype
            )  # Ensure centroids match data type

            for _ in range(self.max_iters):
                # 2. Assignment Step
                distances = self._compute_distances(X)
                labels = np.argmin(distances, axis=1)

                # 3. Update Step
                new_centroids = np.array(
                    [X[labels == k].mean(axis=0) for k in range(self.n_clusters)]
                )

                # Robustness: Handle empty clusters by keeping the old centroid
                nan_mask = np.isnan(new_centroids).any(axis=1)
                if np.any(nan_mask):
                    new_centroids[nan_mask] = self.centroids[nan_mask]

                # 4. Check for convergence (use np.allclose for float comparison)
                if np.allclose(self.centroids, new_centroids):
                    break

                self.centroids = new_centroids

            # The final self.centroids array has the shape (K, 3)
            return self

        def predict(self, X):
            if self.centroids is None:
                raise AttributeError(
                    "Model has not been fitted yet. Call .fit() first."
                )

            distances = self._compute_distances(X)
            return np.argmin(distances, axis=1)

        def _compute_distances(self, X):
            # Broadcasting logic to compute the squared Euclidean distance:
            # X[:, np.newaxis, :] becomes (N, 1, 3)
            # self.centroids[np.newaxis, :, :] becomes (1, K, 3)
            # Result of subtraction is (N, K, 3), sum(axis=2) results in (N, K)
            diff = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
            diff_sq = diff**2
            diff_sq_sum = diff_sq.sum(axis=2)
            return np.sqrt(diff_sq_sum)
    return (KMeans_NP,)


@app.cell
def _(np, plt):
    import matplotlib.patches as patches


    def plot_output(
        original: np.ndarray,
        dominant_colors: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Plots the original image, a palette of dominant colors, and a pie chart
        showing their dominance percentage, with borders added.

        Args:
            original (np.ndarray): The original image (e.g., W x H x 3).
            dominant_colors (np.ndarray): The K cluster centers (K x 3).
            labels (np.ndarray): The cluster assignment for each pixel (N).
        """

        K = dominant_colors.shape[0]

        # CORRECTION: Use np.unique to get counts
        cluster_indices, counts = np.unique(labels, return_counts=True)

        # 1. Create the figure. Use 3 fixed subplots (1 row, 3 columns)
        plt.figure(figsize=(21, 6))

        # --- Plot 1: Original Image (Position 1) ---
        ax_original = plt.subplot(1, 3, 1)  # Corrected: Use 3 columns
        ax_original.imshow(original)
        ax_original.set_title("Original Image")
        ax_original.axis("off")

        # --- Plot 2: Dominant Colors Bar (Position 2) ---
        # Create the color bar strip based on dominance
        color_bar = np.zeros((100, 500, 3), dtype=np.uint8)
        total_pixels = np.sum(counts)

        # Sort by dominance (counts)
        sorted_indices = np.argsort(-counts)

        start_x = 0
        for i in sorted_indices:
            color = dominant_colors[i]
            width = int(counts[i] / total_pixels * 500)

            # Draw the color segment
            color_bar[:, start_x : start_x + width] = color
            start_x += width

        ax_colorbar = plt.subplot(1, 3, 2)  # Corrected: Use 3 columns
        ax_colorbar.imshow(color_bar)
        ax_colorbar.set_title(f"Top {K} Dominant Colors (Palette)")
        ax_colorbar.axis("off")

        # BORDER FOR COLOR BAR
        # Draw a black rectangle around the image data
        rect = patches.Rectangle(
            (0, 0),
            500,
            100,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            clip_on=False,
        )
        ax_colorbar.add_patch(rect)

        # --- Plot 3: Pie Chart of Dominance (Position 3) ---
        ax_pie = plt.subplot(1, 3, 3)  # Corrected: Use 3 columns

        # Scale colors to 0-1 range for matplotlib pie chart wedge colors
        wedge_colors = dominant_colors[sorted_indices] / 255.0

        # BORDER FOR PIE CHART WEDGES
        wedges, texts, autotexts = ax_pie.pie(
            counts[sorted_indices],
            colors=wedge_colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={
                "linewidth": 1,
                "edgecolor": "black",
            },  # The border property
        )

        ax_pie.set_title("Color Dominance (%)")
        ax_pie.axis("equal")  # Ensures pie is a circle

        plt.tight_layout()
        return plt.gcf()
    return (plot_output,)


@app.cell
def _(
    BytesIO,
    Image,
    KMeans_NP,
    cluster_count,
    img_file,
    max_iters,
    mo,
    np,
    plot_output,
):
    mo.stop(len(img_file.value) < 1)

    image = Image.open(BytesIO(img_file.value[0].contents))
    shape = image.size
    image_array = np.asarray(image)
    # if shape larger than 1000x1000, resize while maintaining aspect ratio and 3 channels
    if shape[0] > 1000 or shape[1] > 1000:
        aspect_ratio = shape[0] / shape[1]
        if aspect_ratio > 1:
            new_width = 1000
            new_height = int(1000 / aspect_ratio)
        else:
            new_height = 1000
            new_width = int(1000 * aspect_ratio)
        image = image.resize((new_width, new_height))
        image_array = np.asarray(image)

    X = image_array.reshape((-1, 3))

    kmeans = KMeans_NP(
        n_clusters=cluster_count.value,
        random_state=42,
        max_iters=max_iters.value,
    )
    kmeans.fit(X)

    mo.vstack(
        [
            mo.md("## Dominant Colors"),
            plot_output(
                image_array,
                kmeans.cluster_centers_.astype(np.uint8),
                kmeans.predict(X),
            ),
        ]
    )
    return (kmeans,)


@app.cell
def _(kmeans, mo, np, plt):
    def clamp(x):
        return max(0, min(x, 255))


    def rgb_to_hex(rgb):
        r, g, b = rgb
        return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))


    _colors = kmeans.cluster_centers_.astype(np.uint8)


    def to_md():
        for _i, color in enumerate(_colors, 1):
            output = ""
            output += f"## **Color {_i}**\n"
            output += f"### R: {color[0]}, G: {color[1]}, B: {color[2]}\n"
            output += f"### **Hex**: {rgb_to_hex(color)}\n<br>\n"
            plt.figure(figsize=(5, 5))
            plt.imshow([[color]])
            plt.axis("off")
            yield mo.vstack([plt.gcf(), mo.md(output)])


    mo.hstack(to_md())
    return


if __name__ == "__main__":
    app.run()
