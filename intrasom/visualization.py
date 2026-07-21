import os
import glob
from io import BytesIO
from math import pi
from textwrap import fill
from importlib import resources

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from tqdm.auto import tqdm

import plotly.graph_objs as go
from scipy.ndimage import rotate
from skimage.transform import resize
from sklearn.cluster import MiniBatchKMeans


class PlotFactory:
    """
    Visualization utilities for IntraSOM.

    Public SOM map convention
    -------------------------
    mapsize = (columns, rows)

    NumPy matrices are always stored as
    -------------------------------
    shape = (rows, columns)

    Linear neuron indexing is row-major
    ----------------------------------
    row = node_index // columns
    col = node_index % columns

    Both hexagonal (``hexa``) and rectangular (``rect``) lattices are
    supported. Plot geometry is generated from the lattice type rather
    than assuming a hexagonal map.
    """

    _VALID_LATTICES = {"hexa", "rect"}
    _VALID_MAPSHAPES = {"planar", "toroid"}

    def __init__(self, som_object):
        self.name = som_object.name
        self.codebook = som_object.codebook.matrix

        # Public convention: mapsize = (columns, rows)
        self.mapsize = tuple(int(v) for v in som_object.mapsize)
        if len(self.mapsize) != 2:
            raise ValueError(
                "mapsize must contain exactly two values: (columns, rows)."
            )

        self.cols, self.rows = self.mapsize

        if self.cols <= 0 or self.rows <= 0:
            raise ValueError("mapsize values must be greater than zero.")

        self.lattice = getattr(som_object, "lattice", "hexa")
        self.mapshape = getattr(som_object, "mapshape", "planar")

        if self.lattice not in self._VALID_LATTICES:
            raise ValueError(
                f"Unsupported lattice {self.lattice!r}. "
                "Accepted values are 'hexa' and 'rect'."
            )

        if self.mapshape not in self._VALID_MAPSHAPES:
            raise ValueError(
                f"Unsupported mapshape {self.mapshape!r}. "
                "Accepted values are 'planar' and 'toroid'."
            )

        self.bmus = np.asarray(som_object._bmu[0], dtype=int)
        self.neuron_matrix = np.asarray(som_object.neuron_matrix)
        self.component_names = np.asarray(som_object._component_names)
        self.sample_names = np.asarray(som_object._sample_names)
        self.unit_names = np.asarray(som_object._unit_names)
        self.rep_sample = som_object.rep_sample
        self.data_denorm = np.asarray(
            som_object.denorm_data(som_object._data)
        )
        self.data_proj_norm = som_object.data_proj_norm

        # Use the SOM implementation as the single source of truth for
        # U-Matrix distances.
        self.build_umatrix = som_object.build_umatrix

        if self.codebook.shape[0] != self.cols * self.rows:
            raise ValueError(
                "The codebook number of neurons is inconsistent with mapsize. "
                f"codebook={self.codebook.shape[0]}, "
                f"mapsize={self.mapsize} -> {self.cols * self.rows} neurons."
            )

        # Geometry cache avoids rebuilding coordinates for every plot.
        self._geometry_cache = {}

        # Load watermark once. Failure to locate the image must not prevent
        # plotting from working.
        self.foot = None
        try:
            image_file = resources.files("intrasom") / "images" / "foot.jpg"
            with Image.open(image_file) as img:
                self.foot = img.copy()
        except Exception:
            self.foot = None

    # ------------------------------------------------------------------
    # GEOMETRY HELPERS
    # ------------------------------------------------------------------

    def _node_coordinates(self):
        """
        Return node coordinates as two arrays with shape (rows, columns).

        The map API uses (columns, rows), while NumPy indexing always uses
        [row, column].
        """
        cache_key = ("node_coordinates", self.lattice, self.cols, self.rows)

        if cache_key in self._geometry_cache:
            return self._geometry_cache[cache_key]

        if self.lattice == "hexa":
            coordinates = self.generate_hex_lattice(
                self.cols,
                self.rows
            )
        else:
            coordinates = self.generate_rec_lattice(
                self.cols,
                self.rows
            )

        xx = coordinates[:, 0].reshape(
            self.rows,
            self.cols
        )
        yy = coordinates[:, 1].reshape(
            self.rows,
            self.cols
        )

        self._geometry_cache[cache_key] = (xx, yy)
        return xx, yy

    @staticmethod
    def _safe_norm(values):
        """
        Build a Matplotlib Normalize that remains valid for constant data.
        """
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]

        if finite.size == 0:
            return mpl.colors.Normalize(vmin=0.0, vmax=1.0)

        vmin = float(np.min(finite))
        vmax = float(np.max(finite))

        if np.isclose(vmin, vmax):
            pad = 0.5 if np.isclose(vmin, 0.0) else abs(vmin) * 0.01
            if np.isclose(pad, 0.0):
                pad = 0.5
            vmin -= pad
            vmax += pad

        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    @staticmethod
    def _polygon_vertices(
        centers,
        lattice,
        sizes=1.0,
    ):
        """
        Vectorized polygon generation.

        Parameters
        ----------
        centers : array-like, shape (n, 2)
            Cell centers.
        lattice : {"hexa", "rect"}
            Shape to generate.
        sizes : float or array-like
            Relative cell scale. A value of 1 reproduces the base SOM cell.
        """
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        n = centers.shape[0]

        sizes = np.asarray(sizes, dtype=float)
        if sizes.ndim == 0:
            sizes = np.full(n, float(sizes), dtype=float)
        else:
            sizes = np.broadcast_to(sizes.reshape(-1), (n,)).astype(float)

        if lattice == "hexa":
            # Preserve the historical RegularPolygon orientation used by
            # visualization.py while generating all polygons vectorially.
            angles = (
                np.pi / 2.0
                + np.arange(6, dtype=float) * (2.0 * np.pi / 6.0)
            )
            radius = (1.0 / np.sqrt(3.0)) * sizes
            offsets = np.stack(
                [np.cos(angles), np.sin(angles)],
                axis=1
            )
            vertices = (
                centers[:, None, :]
                + radius[:, None, None] * offsets[None, :, :]
            )
            return vertices

        # Rectangular lattice: one square per neuron at unit spacing.
        half = 0.5 * sizes
        base = np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            dtype=float,
        )
        vertices = (
            centers[:, None, :]
            + half[:, None, None] * base[None, :, :]
        )
        return vertices

    def _add_value_cells(
        self,
        ax,
        centers,
        values,
        *,
        cmap,
        norm,
        lattice=None,
        sizes=1.0,
        edgecolors="none",
        linewidths=0.0,
        alpha=1.0,
        zorder=1,
    ):
        """
        Add many colored SOM cells in one PolyCollection.

        This is substantially faster than adding one Matplotlib patch at a
        time, especially for large maps.
        """
        lattice = self.lattice if lattice is None else lattice

        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        values = np.asarray(values, dtype=float).reshape(-1)

        if centers.shape[0] != values.size:
            raise ValueError(
                "centers and values must contain the same number of cells."
            )

        valid = np.isfinite(values)

        if not np.any(valid):
            return None

        centers = centers[valid]
        values = values[valid]

        if np.ndim(sizes) == 0:
            filtered_sizes = sizes
        else:
            filtered_sizes = np.asarray(sizes).reshape(-1)[valid]

        vertices = self._polygon_vertices(
            centers,
            lattice,
            filtered_sizes,
        )

        collection = PolyCollection(
            vertices,
            cmap=cmap,
            norm=norm,
            edgecolors=edgecolors,
            linewidths=linewidths,
            alpha=alpha,
            zorder=zorder,
        )
        collection.set_array(values)

        ax.add_collection(collection)
        return collection

    def _add_solid_cells(
        self,
        ax,
        centers,
        *,
        facecolors,
        edgecolors=None,
        linewidths=0.0,
        alpha=1.0,
        sizes=1.0,
        lattice=None,
        zorder=3,
    ):
        """
        Add many solid-color SOM cells in one PolyCollection.
        """
        lattice = self.lattice if lattice is None else lattice
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)

        if centers.size == 0:
            return None

        vertices = self._polygon_vertices(
            centers,
            lattice,
            sizes,
        )

        if edgecolors is None:
            edgecolors = facecolors

        collection = PolyCollection(
            vertices,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=linewidths,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_collection(collection)
        return collection

    def _set_map_limits(
        self,
        ax,
        centers,
        *,
        pad=0.75,
        invert_y=True,
    ):
        """
        Set axis limits from actual geometry instead of hard-coded formulas.

        This prevents row/column mistakes on non-square maps.
        """
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)

        if centers.size == 0:
            return

        xmin = float(np.nanmin(centers[:, 0])) - pad
        xmax = float(np.nanmax(centers[:, 0])) + pad
        ymin = float(np.nanmin(centers[:, 1])) - pad
        ymax = float(np.nanmax(centers[:, 1])) + pad

        ax.set_xlim(xmin, xmax)

        if invert_y:
            ax.set_ylim(ymax, ymin)
        else:
            ax.set_ylim(ymin, ymax)

        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

    def _umatrix_geometry(
        self,
        um,
        umat,
        *,
        include_toroid_wrap=False,
    ):
        """
        Convert reduced and expanded U-Matrix arrays into drawable cells.

        ``build_umatrix`` is expected to return:
            hexa -> expanded shape (rows, cols, 6)
            rect -> expanded shape (rows, cols, 8)

        Only one copy of each undirected relation is drawn:
            hexa -> right, down-right, down-left
            rect -> right, down-right, down, down-left

        Important for toroidal maps
        ---------------------------
        A flat 2-D U-Matrix must not draw wrap-around relations as ordinary
        cells *outside* the fundamental map rectangle. Doing so creates the
        visual artifact where every interstitial/alternate row appears shifted
        left or right.

        Therefore:
            include_toroid_wrap=False
                Used by the ordinary 2-D U-Matrix. Wrap-crossing edge cells
                are omitted from the flat representation.

            include_toroid_wrap=True
                Used only by the torus texture renderer, where periodic
                relations are required and the geometry is tiled/cropped.
        """
        um = np.asarray(
            um,
            dtype=float,
        )
        umat = np.asarray(
            umat,
            dtype=float,
        )

        expected_umat = (
            self.rows,
            self.cols,
        )

        if umat.shape != expected_umat:
            raise ValueError(
                "Reduced U-Matrix has an invalid shape. "
                f"Received {umat.shape}, expected {expected_umat}."
            )

        expected_neighbors = (
            6
            if self.lattice == "hexa"
            else 8
        )

        expected_um = (
            self.rows,
            self.cols,
            expected_neighbors,
        )

        if um.shape != expected_um:
            raise ValueError(
                "Expanded U-Matrix has an invalid shape. "
                f"Received {um.shape}, expected {expected_um}. "
                "Check that build_umatrix supports the selected lattice."
            )

        xx, yy = self._node_coordinates()

        node_centers = np.column_stack(
            [
                (2.0 * xx).ravel(),
                (2.0 * yy).ravel(),
            ]
        )

        node_values = umat.ravel()

        row_grid, col_grid = np.indices(
            (
                self.rows,
                self.cols,
            ),
            dtype=int,
        )

        if self.lattice == "hexa":

            # Expanded U-Matrix slots:
            #   0 = right
            #   1 = down-right
            #   2 = down-left
            #
            # The actual column step depends on odd-r row parity, while the
            # geometric midpoint offsets remain +/-0.5 after the node
            # coordinates are doubled.
            offsets = np.array(
                [
                    [1.0, 0.0],
                    [
                        0.5,
                        np.sqrt(3.0) / 2.0,
                    ],
                    [
                        -0.5,
                        np.sqrt(3.0) / 2.0,
                    ],
                ],
                dtype=float,
            )

            neighbor_slots = (
                0,
                1,
                2,
            )

            even_rows = (
                row_grid % 2
            ) == 0

            neighbor_steps = (
                (
                    np.ones_like(col_grid),
                    np.zeros_like(row_grid),
                ),
                (
                    np.where(
                        even_rows,
                        0,
                        1,
                    ),
                    np.ones_like(row_grid),
                ),
                (
                    np.where(
                        even_rows,
                        -1,
                        0,
                    ),
                    np.ones_like(row_grid),
                ),
            )

        else:

            # Expanded rectangular U-Matrix slots:
            #   0 = right
            #   1 = down-right
            #   2 = down
            #   3 = down-left
            offsets = np.array(
                [
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [-1.0, 1.0],
                ],
                dtype=float,
            )

            neighbor_slots = (
                0,
                1,
                2,
                3,
            )

            neighbor_steps = (
                (
                    np.ones_like(col_grid),
                    np.zeros_like(row_grid),
                ),
                (
                    np.ones_like(col_grid),
                    np.ones_like(row_grid),
                ),
                (
                    np.zeros_like(col_grid),
                    np.ones_like(row_grid),
                ),
                (
                    -np.ones_like(col_grid),
                    np.ones_like(row_grid),
                ),
            )

        edge_centers = []
        edge_values = []

        for (
            offset,
            slot,
            (
                step_x,
                step_y,
            ),
        ) in zip(
            offsets,
            neighbor_slots,
            neighbor_steps,
        ):

            values = np.array(
                um[:, :, slot],
                dtype=float,
                copy=True,
            )

            # In a planar U-Matrix, build_umatrix already returns NaN for
            # relations outside the map. For a toroidal map those relations
            # are finite because the neighbor wraps to the opposite border.
            #
            # In the ordinary 2-D view, suppress only those wrap-crossing
            # relations so they are not drawn one cell outside the map.
            if (
                self.mapshape == "toroid"
                and not include_toroid_wrap
            ):
                neighbor_rows = (
                    row_grid
                    + step_y
                )
                neighbor_cols = (
                    col_grid
                    + step_x
                )

                inside_flat_domain = (
                    (neighbor_rows >= 0)
                    & (
                        neighbor_rows
                        < self.rows
                    )
                    & (neighbor_cols >= 0)
                    & (
                        neighbor_cols
                        < self.cols
                    )
                )

                values[
                    ~inside_flat_domain
                ] = np.nan

            edge_centers.append(
                node_centers
                + offset
            )

            edge_values.append(
                values.ravel()
            )

        edge_centers = np.vstack(
            edge_centers
        )

        edge_values = np.concatenate(
            edge_values
        )

        return (
            node_centers,
            node_values,
            edge_centers,
            edge_values,
        )

    def _render_figure_to_array(self, fig):
        """
        Render a Matplotlib figure directly to an RGB ndarray.

        Avoids the previous save-to-disk/read-from-disk roundtrip.
        """
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        rgba = np.asarray(canvas.buffer_rgba())
        return np.ascontiguousarray(rgba[:, :, :3])

    def _plot_umatrix_texture(
        self,
        *,
        um,
        umat,
        hits=True,
        cmap,
        norm,
        figsize=(10, 10),
    ):
        """
        Render a seamless periodic U-Matrix texture for ``plot_torus``.

        The geometry is tiled in a 3x3 neighborhood and cropped to one exact
        map period. This is robust for non-square maps and for both lattices.
        """
        (
            node_centers,
            node_values,
            edge_centers,
            edge_values,
        ) = self._umatrix_geometry(
            um,
            umat,
            include_toroid_wrap=True,
        )

        if self.lattice == "hexa":
            period_x = 2.0 * self.cols
            period_y = np.sqrt(3.0) * self.rows
            crop_x0 = -1.0
            crop_y0 = -np.sqrt(3.0) / 2.0
        else:
            period_x = 2.0 * self.cols
            period_y = 2.0 * self.rows
            crop_x0 = -1.0
            crop_y0 = -1.0

        translations = np.array(
            [
                [dx * period_x, dy * period_y]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ],
            dtype=float,
        )

        tiled_nodes = np.vstack(
            [node_centers + shift for shift in translations]
        )
        tiled_node_values = np.tile(
            node_values,
            len(translations)
        )

        tiled_edges = np.vstack(
            [edge_centers + shift for shift in translations]
        )
        tiled_edge_values = np.tile(
            edge_values,
            len(translations)
        )

        # Render the texture with the same physical aspect ratio as one
        # periodic map tile. A square Matplotlib canvas combined with
        # ``ax.set_aspect("equal")`` creates white letterbox margins for
        # non-square maps; those margins become a white sector on the torus.
        requested_w = float(figsize[0])
        requested_h = float(figsize[1])
        max_side = max(requested_w, requested_h)

        if period_x >= period_y:
            texture_figsize = (
                max_side,
                max_side * period_y / period_x,
            )
        else:
            texture_figsize = (
                max_side * period_x / period_y,
                max_side,
            )

        fig, ax = plt.subplots(
            figsize=texture_figsize,
            dpi=180,
        )
        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        self._add_value_cells(
            ax,
            tiled_nodes,
            tiled_node_values,
            cmap=cmap,
            norm=norm,
        )
        self._add_value_cells(
            ax,
            tiled_edges,
            tiled_edge_values,
            cmap=cmap,
            norm=norm,
        )

        if hits:
            hit_sizes = self.hits_dictionary
            hit_nodes = np.array(
                sorted(hit_sizes.keys()),
                dtype=int,
            )

            if hit_nodes.size:
                valid = (
                    (hit_nodes >= 0)
                    & (hit_nodes < node_centers.shape[0])
                )
                hit_nodes = hit_nodes[valid]

                hit_centers = node_centers[hit_nodes]
                hit_scales = np.array(
                    [hit_sizes[int(i)] for i in hit_nodes],
                    dtype=float,
                )

                tiled_hit_centers = np.vstack(
                    [hit_centers + shift for shift in translations]
                )
                tiled_hit_scales = np.tile(
                    hit_scales,
                    len(translations)
                )

                self._add_solid_cells(
                    ax,
                    tiled_hit_centers,
                    facecolors="white",
                    edgecolors="lightgray",
                    linewidths=0.4,
                    sizes=tiled_hit_scales,
                    zorder=4,
                )

        ax.set_xlim(
            crop_x0,
            crop_x0 + period_x,
        )
        ax.set_ylim(
            crop_y0 + period_y,
            crop_y0,
        )
        ax.set_axis_off()
        ax.set_facecolor("none")
        fig.patch.set_alpha(0.0)
        fig.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1,
        )

        image = self._render_figure_to_array(fig)
        plt.close(fig)

        return image

    # ------------------------------------------------------------------
    # U-MATRIX
    # ------------------------------------------------------------------

    def plot_umatrix(
        self,
        figsize=(10, 10),
        hits=True,
        title="U-Matrix",
        title_size=40,
        title_pad=25,
        legend_title="Distance",
        legend_title_size=25,
        legend_ticks_size=20,
        save=True,
        watermark_neurons=False,
        watermark_neurons_alfa=0.5,
        neurons_fontsize=7,
        file_name=None,
        file_path=False,
        resume=False,
        label_plot=False,
        label_plot_name=None,
        project_samples_label=None,
        samples_label=False,
        samples_label_index=None,
        samples_label_fontsize=8,
        save_labels_rep=False,
        label_title_xy=(0, 0.5),
        log=False,
        cmap="jet",
    ):
        """
        Plot the SOM U-Matrix for hexagonal or rectangular lattices.

        ``mapsize`` is interpreted strictly as ``(columns, rows)``.
        Internally, all matrices are reshaped as ``(rows, columns)``.

        For rectangular lattices, neuron cells and inter-neuron distance
        cells are drawn as squares, preserving the same visual logic used
        by the hexagonal U-Matrix.

        ``resume=True`` returns a seamless RGB texture used by
        :meth:`plot_torus`.
        """
        if file_name is None:
            file_name = f"U_Matrix_{self.name}"

        um = np.asarray(
            self.build_umatrix(
                expanded=True,
                log=log,
            ),
            dtype=float,
        )
        umat = np.asarray(
            self.build_umatrix(
                expanded=False,
                log=log,
            ),
            dtype=float,
        )

        all_values = np.concatenate(
            [
                um[np.isfinite(um)],
                umat[np.isfinite(umat)],
            ]
        )
        norm = self._safe_norm(all_values)
        # IntraSOM visualization standard: always use JET.
        # ``cmap`` is retained only for backward API compatibility.
        cmap_obj = mpl.colormaps["jet"]

        if resume:
            image = self._plot_umatrix_texture(
                um=um,
                umat=umat,
                hits=hits,
                cmap=cmap_obj,
                norm=norm,
                figsize=figsize,
            )

            if save:
                path = (
                    str(file_path)
                    if file_path
                    else "Plots/U_matrix"
                )
                os.makedirs(path, exist_ok=True)

                Image.fromarray(image).save(
                    os.path.join(
                        path,
                        f"{file_name}_texture.jpg",
                    ),
                    quality=95,
                )

            return image

        (
            node_centers,
            node_values,
            edge_centers,
            edge_values,
        ) = self._umatrix_geometry(
            um,
            umat,
            include_toroid_wrap=False,
        )

        fig = plt.figure(
            figsize=figsize,
            dpi=300,
        )

        gs = gridspec.GridSpec(
            100,
            100,
            figure=fig,
        )
        ax = fig.add_subplot(
            gs[:95, 0:90]
        )
        ax.set_aspect("equal")

        # Draw all U-Matrix cells in two vectorized collections.
        self._add_value_cells(
            ax,
            node_centers,
            node_values,
            cmap=cmap_obj,
            norm=norm,
            zorder=1,
        )
        self._add_value_cells(
            ax,
            edge_centers,
            edge_values,
            cmap=cmap_obj,
            norm=norm,
            zorder=1,
        )

        # Optional neuron-number watermark.
        if watermark_neurons:
            self._add_solid_cells(
                ax,
                node_centers,
                facecolors="white",
                edgecolors="black",
                linewidths=0.6,
                alpha=watermark_neurons_alfa,
                sizes=2.0,
                zorder=2,
            )

            for node_ind, (x, y) in enumerate(
                node_centers,
                start=1,
            ):
                ax.text(
                    x,
                    y,
                    str(node_ind),
                    size=neurons_fontsize,
                    ha="center",
                    va="center",
                    color="black",
                    zorder=5,
                )

        # Hits.
        if hits:
            hit_sizes_dict = self.hits_dictionary
            hit_nodes = np.array(
                sorted(hit_sizes_dict.keys()),
                dtype=int,
            )

            if hit_nodes.size:
                valid = (
                    (hit_nodes >= 0)
                    & (hit_nodes < node_centers.shape[0])
                )
                hit_nodes = hit_nodes[valid]

                hit_centers = node_centers[hit_nodes]
                hit_sizes = np.array(
                    [
                        hit_sizes_dict[int(node)]
                        for node in hit_nodes
                    ],
                    dtype=float,
                )

                facecolors = np.full(
                    hit_nodes.size,
                    "white",
                    dtype=object,
                )

                if label_plot:
                    if label_plot_name is None:
                        raise ValueError(
                            "label_plot_name must be provided when "
                            "label_plot=True."
                        )

                    try:
                        ind_var = list(
                            self.component_names
                        ).index(label_plot_name)
                    except ValueError as exc:
                        raise ValueError(
                            f"Unknown component {label_plot_name!r}."
                        ) from exc

                    bool_val = (
                        self.data_denorm[:, ind_var] >= 0.5
                    ).astype(int)

                    # Majority label per BMU. This avoids repeatedly scanning
                    # all samples inside the plotting loop.
                    for pos, node in enumerate(hit_nodes):
                        values = bool_val[
                            self.bmus == node
                        ]

                        if values.size:
                            # Binary mode; ties resolve to 0, matching a
                            # conservative "absence" interpretation.
                            ones = int(np.count_nonzero(values))
                            zeros = int(values.size - ones)
                            facecolors[pos] = (
                                "white"
                                if ones > zeros
                                else "black"
                            )

                self._add_solid_cells(
                    ax,
                    hit_centers,
                    facecolors=facecolors,
                    edgecolors=facecolors,
                    linewidths=1.1,
                    sizes=hit_sizes,
                    zorder=4,
                )

        # Selected sample labels.
        if samples_label:
            if project_samples_label is not None:
                samples_label_names = np.asarray(
                    project_samples_label.index.tolist()
                )
                som_bmus = (
                    np.asarray(
                        project_samples_label.BMU.values,
                        dtype=int,
                    )
                    - 1
                )
                representative = self.rep_sample(
                    project=project_samples_label
                )
            else:
                if isinstance(
                    samples_label_index,
                    str,
                ) and samples_label_index.lower() == "all":
                    selected_indices = np.arange(
                        len(self.sample_names),
                        dtype=int,
                    )
                else:
                    if samples_label_index is None:
                        raise ValueError(
                            "samples_label_index must be provided when "
                            "samples_label=True, or use 'all'."
                        )
                    selected_indices = np.asarray(
                        samples_label_index,
                        dtype=int,
                    ).reshape(-1)

                samples_label_names = self.sample_names[
                    selected_indices
                ]
                som_bmus = self.bmus[
                    selected_indices
                ]
                representative = self.rep_sample()

            selected_names_set = set(
                map(str, samples_label_names)
            )
            selected_bmus = set(
                int(v) + 1
                for v in np.asarray(som_bmus).reshape(-1)
            )

            representative = {
                key: value
                for key, value in representative.items()
                if key in selected_bmus
            }

            if save_labels_rep:
                os.makedirs(
                    "Results",
                    exist_ok=True,
                )
                with open(
                    "Results/Representative_samples_umatrix.txt",
                    "w",
                    encoding="utf-8",
                ) as file:
                    for key, value in representative.items():
                        if isinstance(value, list):
                            filtered = [
                                item
                                for item in value
                                if str(item) in selected_names_set
                            ]
                            if not filtered:
                                continue
                            out_value = ", ".join(
                                map(str, filtered)
                            )
                        else:
                            if str(value) not in selected_names_set:
                                continue
                            out_value = str(value)

                        file.write(
                            f"BMU {key}: {out_value}\n"
                        )

            # One annotation per selected BMU.
            for bmu_1based, rep_value in representative.items():
                node = int(bmu_1based) - 1

                if not (
                    0 <= node < node_centers.shape[0]
                ):
                    continue

                if isinstance(rep_value, list):
                    selected = [
                        item
                        for item in rep_value
                        if str(item) in selected_names_set
                    ]
                    if not selected:
                        continue

                    # Use the earliest representative among the selected
                    # samples.
                    indices = [
                        rep_value.index(item)
                        for item in selected
                    ]
                    best_local = int(np.argmin(indices))
                    sample_name = selected[best_local]
                    idx_sample = indices[best_local]
                    rep_name = (
                        f"{sample_name}"
                        f"({idx_sample + 1}/{len(rep_value)})"
                    )
                else:
                    if str(rep_value) not in selected_names_set:
                        continue
                    rep_name = str(rep_value)

                center = node_centers[
                    node
                ]

                self._add_solid_cells(
                    ax,
                    center.reshape(1, 2),
                    facecolors="black",
                    edgecolors="white",
                    linewidths=1.1,
                    sizes=0.5,
                    zorder=5,
                )

                vertical_offset = (
                    1.5 / np.sqrt(3.0)
                    if self.lattice == "hexa"
                    else 0.9
                )

                ax.text(
                    center[0],
                    center[1] + vertical_offset,
                    rep_name,
                    size=samples_label_fontsize,
                    ha="center",
                    va="top",
                    color="black",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "black",
                        "boxstyle": "round",
                    },
                    zorder=6,
                )

        # Limits are derived from the flat-domain geometry only.
        # Toroidal wrap-crossing cells were filtered above, preventing
        # alternate/interstitial rows from extending beyond the map borders.
        all_centers = np.vstack(
            [
                node_centers,
                edge_centers[
                    np.isfinite(edge_values)
                ],
            ]
        )
        # Base U-Matrix cells need ~0.6 map units of padding. Hits can be
        # scaled up to 2x, so their vertices may extend farther than the
        # U-Matrix centers, especially at the outer border.
        map_pad = 0.8

        if hits:
            hit_sizes_for_limits = self.hits_dictionary

            if hit_sizes_for_limits:
                max_hit_scale = float(
                    max(
                        hit_sizes_for_limits.values()
                    )
                )

                if self.lattice == "hexa":
                    # Pointy-top hexagon: maximum vertical radius.
                    overlay_extent = (
                        max_hit_scale
                        / np.sqrt(3.0)
                    )
                else:
                    # Square: half side length.
                    overlay_extent = (
                        0.5
                        * max_hit_scale
                    )

                map_pad = max(
                    map_pad,
                    overlay_extent + 0.25,
                )

        if samples_label:
            # Leave room for annotation boxes close to the outer border.
            map_pad = max(
                map_pad,
                1.35,
            )

        self._set_map_limits(
            ax,
            all_centers,
            pad=map_pad,
            invert_y=True,
        )

        ax.set_title(
            fill(str(title), 30),
            size=title_size,
            pad=title_pad,
        )

        # Colorbar uses the same colormap as the map itself.
        ax2 = fig.add_subplot(
            gs[30:70, 95:98]
        )

        scalar_mappable = mpl.cm.ScalarMappable(
            norm=norm,
            cmap=cmap_obj,
        )
        cb1 = fig.colorbar(
            scalar_mappable,
            cax=ax2,
            orientation="vertical",
        )
        cb1.ax.tick_params(
            labelsize=legend_ticks_size
        )
        cb1.set_label(
            fill(str(legend_title), 20),
            size=legend_title_size,
            labelpad=20,
        )
        cb1.ax.yaxis.label.set_position(
            label_title_xy
        )

        self._add_watermark_subplot(
            fig,
            gs,
        )

        if save:
            if file_path:
                path = str(file_path)
            else:
                path = "Plots/U_matrix"

            os.makedirs(
                path,
                exist_ok=True,
            )

            if hits:
                if label_plot:
                    suffix = "_with_hits_label"
                elif watermark_neurons:
                    suffix = "_with_hits_watermarkneurons"
                else:
                    suffix = "_with_hits"
            else:
                if label_plot:
                    suffix = "_with_label"
                elif watermark_neurons:
                    suffix = "_watermarkneurons"
                else:
                    suffix = ""

            fig.savefig(
                os.path.join(
                    path,
                    f"{file_name}{suffix}.jpg",
                ),
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()
        return fig

    # ------------------------------------------------------------------
    # COMPONENT PLOTS
    # ------------------------------------------------------------------

    def _resolve_component(
        self,
        component_name,
    ):
        """
        Resolve a component index/name and reshape values correctly.

        mapsize = (columns, rows)
        ndarray shape = (rows, columns)
        """
        if isinstance(
            component_name,
            (int, np.integer),
        ):
            index = int(component_name)

            if not (
                0 <= index < self.neuron_matrix.shape[1]
            ):
                raise IndexError(
                    f"Component index {index} is out of bounds."
                )
        elif isinstance(
            component_name,
            str,
        ):
            try:
                index = list(
                    self.component_names
                ).index(component_name)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown component {component_name!r}."
                ) from exc
        else:
            raise TypeError(
                "component_name must be a component name or integer index."
            )

        values = self.neuron_matrix[
            :,
            index,
        ].reshape(
            self.rows,
            self.cols,
        )

        return (
            index,
            self.component_names[index],
            values,
        )

    def component_plot(
        self,
        component_name=0,
        figsize=(10, 10),
        title=None,
        full_title=False,
        title_size=30,
        title_pad=25,
        legend_title=False,
        legend_pad=0,
        label_title_xy=(0, 0.5),
        legend_title_size=24,
        legend_ticks_size=20,
        save=False,
        file_name=None,
        file_path=False,
        collage=False,
        cmap="jet",
    ):
        """
        Plot one trained component for hexagonal or rectangular lattices.

        The map geometry is generated using ``mapsize=(columns, rows)`` and
        values are reshaped as ``(rows, columns)``.
        """
        (
            index,
            var_name,
            bmu_var,
        ) = self._resolve_component(
            component_name
        )

        if not legend_title:
            legend_title = self.unit_names[
                index
            ]

        xx, yy = self._node_coordinates()
        centers = np.column_stack(
            [
                xx.ravel(),
                yy.ravel(),
            ]
        )
        values = bmu_var.ravel()

        norm = self._safe_norm(values)
        # IntraSOM visualization standard: always use JET.
        # ``cmap`` is retained only for backward API compatibility.
        cmap_obj = mpl.colormaps["jet"]

        fig = plt.figure(
            figsize=figsize,
            dpi=300,
        )
        gs = gridspec.GridSpec(
            100,
            20,
            figure=fig,
        )
        ax = fig.add_subplot(
            gs[:95, 0:19]
        )
        ax.set_aspect("equal")

        self._add_value_cells(
            ax,
            centers,
            values,
            cmap=cmap_obj,
            norm=norm,
            edgecolors="none",
            zorder=1,
        )

        self._set_map_limits(
            ax,
            centers,
            pad=0.7,
            invert_y=True,
        )

        if full_title:
            name = (
                title
                if title is not None
                else str(var_name)
            )
        else:
            if title is not None:
                name = title
            else:
                split_name = str(
                    var_name
                ).split()
                name = " ".join(
                    split_name[:2]
                )

        ax.set_title(
            fill(str(name), 20),
            size=title_size,
            pad=title_pad,
        )

        ax2 = fig.add_subplot(
            gs[27:70, 19]
        )
        scalar_mappable = mpl.cm.ScalarMappable(
            norm=norm,
            cmap=cmap_obj,
        )
        cb1 = fig.colorbar(
            scalar_mappable,
            cax=ax2,
            orientation="vertical",
        )
        cb1.ax.tick_params(
            labelsize=legend_ticks_size
        )
        cb1.set_label(
            fill(str(legend_title), 15),
            size=legend_title_size,
            labelpad=legend_pad,
            horizontalalignment="right",
            wrap=True,
        )
        cb1.ax.yaxis.label.set_position(
            label_title_xy
        )

        if not collage:
            self._add_watermark_subplot(
                fig,
                gs,
                grid_slice=(slice(95, 100), slice(0, 4)),
            )

        label_name = self._component_label_name(
            var_name,
            full_title=full_title,
        )

        if collage:
            path = (
                "Plots/Component_plots/"
                "Collage/temp"
            )
            os.makedirs(
                path,
                exist_ok=True,
            )
            path_name = self._unique_path(
                os.path.join(
                    path,
                    f"{label_name}.jpg",
                )
            )
            fig.savefig(
                path_name,
                dpi=300,
                bbox_inches="tight",
            )

        if save:
            if file_name is not None:
                label_name = file_name

            path = (
                str(file_path)
                if file_path
                else "Plots/Component_plots"
            )
            os.makedirs(
                path,
                exist_ok=True,
            )
            path_name = self._unique_path(
                os.path.join(
                    path,
                    f"{label_name}.jpg",
                )
            )
            fig.savefig(
                path_name,
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    def multiple_component_plots(
        self,
        wich="all",
        figsize=(10, 10),
        full_title=False,
        title_size=30,
        title_pad=25,
        legend_title="Presence",
        legend_pad=0,
        label_title_xy=(0, 0.5),
        legend_title_size=24,
        legend_ticks_size=20,
        save=True,
        file_path=False,
        collage=False,
        cmap="jet",
    ):
        """
        Plot multiple components.

        ``wich`` may be ``"all"`` or an iterable of component names/indices.
        """
        if isinstance(
            wich,
            str,
        ):
            if wich != "all":
                raise ValueError(
                    "wich must be 'all' or an iterable of component names/indices."
                )
            iterator = list(
                self.component_names
            )
        else:
            iterator = list(wich)

        pbar = tqdm(
            iterator,
            mininterval=1,
        )

        for name in pbar:
            pbar.set_description(
                f"Component: {name}"
            )

            fig = self.component_plot(
                component_name=name,
                figsize=figsize,
                full_title=full_title,
                title_size=title_size,
                title_pad=title_pad,
                legend_title=legend_title,
                legend_title_size=legend_title_size,
                legend_ticks_size=legend_ticks_size,
                legend_pad=legend_pad,
                label_title_xy=label_title_xy,
                save=save,
                file_path=file_path,
                collage=collage,
                cmap=cmap,
            )
            plt.close(fig)

        print("Finished")

    def component_plot_collage(
        self,
        page_size=(2480, 3508),
        grid=(4, 4),
        wich="all",
        figsize=(10, 10),
        full_title=False,
        title_size=30,
        title_pad=25,
        legend_title="Presence",
        legend_title_size=24,
        legend_ticks_size=20,
        legend_pad=0,
        label_title_xy=(0, 0.5),
        file_path=False,
        cmap="jet",
    ):
        """
        Create A4-style collage pages from component plots.

        ``grid`` is interpreted as ``(rows, columns)`` because it describes
        a page layout, not a SOM mapsize.
        """
        grid_rows, grid_cols = (
            int(grid[0]),
            int(grid[1]),
        )

        if (
            grid_rows <= 0
            or grid_cols <= 0
        ):
            raise ValueError(
                "grid values must be greater than zero."
            )

        if isinstance(
            wich,
            str,
        ):
            if wich != "all":
                raise ValueError(
                    "wich must be 'all' or an iterable of component names/indices."
                )
            list_figs = list(
                self.component_names
            )
        else:
            list_figs = list(wich)

        n_components = len(
            list_figs
        )

        if n_components == 0:
            raise ValueError(
                "No components were selected."
            )

        temp_dir = (
            "Plots/Component_plots/"
            "Collage/temp"
        )
        os.makedirs(
            temp_dir,
            exist_ok=True,
        )

        # Avoid mixing stale images from previous runs.
        for old_file in glob.glob(
            os.path.join(
                temp_dir,
                "*.jpg",
            )
        ):
            try:
                os.remove(old_file)
            except OSError:
                pass

        print("Generating maps...")

        self.multiple_component_plots(
            wich=list_figs,
            figsize=figsize,
            full_title=full_title,
            title_size=title_size,
            title_pad=title_pad,
            save=False,
            legend_title=legend_title,
            legend_title_size=legend_title_size,
            legend_ticks_size=legend_ticks_size,
            legend_pad=legend_pad,
            label_title_xy=label_title_xy,
            collage=True,
            cmap=cmap,
        )

        images_path = sorted(
            glob.glob(
                os.path.join(
                    temp_dir,
                    "*.jpg",
                )
            )
        )

        images_per_page = (
            grid_rows * grid_cols
        )
        n_pages = int(
            np.ceil(
                n_components
                / images_per_page
            )
        )

        output_dir = (
            str(file_path)
            if file_path
            else (
                "Plots/Component_plots/"
                "Collage/pages"
            )
        )
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        print("Generating collage...")

        for page_index in range(
            n_pages
        ):
            page = Image.new(
                "RGB",
                page_size,
                "WHITE",
            )

            page_paths = images_path[
                page_index
                * images_per_page:
                (page_index + 1)
                * images_per_page
            ]

            for item_index, img_path in enumerate(
                page_paths
            ):
                with Image.open(
                    img_path
                ) as source:
                    img = source.copy()

                max_width = (
                    page_size[0]
                    / grid_cols
                )
                max_height = (
                    page_size[1]
                    / grid_rows
                )

                img.thumbnail(
                    (
                        int(max_width),
                        int(max_height),
                    )
                )

                page_row = (
                    item_index // grid_cols
                )
                page_col = (
                    item_index % grid_cols
                )

                x_pos = int(
                    page_col
                    * page_size[0]
                    / grid_cols
                )
                y_pos = int(
                    page_row
                    * page_size[1]
                    / grid_rows
                )

                page.paste(
                    img,
                    (
                        x_pos,
                        y_pos,
                    ),
                )

            if self.foot is not None:
                foot = self.foot.copy()
                max_height = int(
                    page_size[1] / 40
                )
                width = int(
                    (
                        foot.width
                        / foot.height
                    )
                    * max_height
                )
                foot.thumbnail(
                    (
                        width,
                        max_height,
                    )
                )
                page.paste(
                    foot,
                    (
                        0,
                        page.height
                        - foot.height,
                    ),
                )

            page.save(
                os.path.join(
                    output_dir,
                    (
                        "Component_plots_"
                        f"collage_page{page_index + 1}.jpg"
                    ),
                ),
                quality=95,
            )

        print("Finished.")

    # ------------------------------------------------------------------
    # BMU TEMPLATE
    # ------------------------------------------------------------------

    def bmu_template(
        self,
        figsize=(10, 10),
        title_size=24,
        fontsize=10,
        alpha_even_rows=0.30,
        alpha_odd_rows=0.80,
        save=False,
        file_name=None,
        file_path=False,
    ):
        """
        Plot the neuron numbering template for either lattice type.
        """
        if file_name is None:
            file_name = self.name

        xx, yy = self._node_coordinates()
        centers = np.column_stack(
            [
                xx.ravel(),
                yy.ravel(),
            ]
        )

        nnodes = (
            self.cols * self.rows
        )
        node_ids = np.arange(
            1,
            nnodes + 1,
            dtype=int,
        )

        row_ids = np.repeat(
            np.arange(self.rows),
            self.cols,
        )
        col_ids = np.tile(
            np.arange(self.cols),
            self.rows,
        )

        # Preserve the alternating historical template appearance while
        # generating colors vectorially.
        norm = mpl.colors.Normalize(
            vmin=0,
            vmax=1,
        )
        cmap = mpl.colormaps["Pastel1"]
        base = np.where(
            col_ids % 2 == 0,
            0.2,
            0.6,
        )
        colors = cmap(
            norm(base)
        )
        alpha_even_rows = float(alpha_even_rows)
        alpha_odd_rows = float(alpha_odd_rows)

        for value, name in (
            (alpha_even_rows, "alpha_even_rows"),
            (alpha_odd_rows, "alpha_odd_rows"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1. Received {value}."
                )

        colors[:, 3] = np.where(
            row_ids % 2 == 0,
            alpha_even_rows,
            alpha_odd_rows,
        )

        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=300,
        )
        ax.set_aspect("equal")

        self._add_solid_cells(
            ax,
            centers,
            facecolors=colors,
            edgecolors="gray",
            linewidths=0.6,
            # Preserve the per-face alpha values already stored in ``colors``.
            # Passing alpha=1.0 here would overwrite all row-specific
            # transparency values inside Matplotlib's PolyCollection.
            alpha=None,
            zorder=1,
        )

        for node_id, (
            x,
            y,
        ) in zip(
            node_ids,
            centers,
        ):
            ax.text(
                x,
                y,
                str(node_id),
                size=fontsize,
                ha="center",
                va="center",
                color="black",
                zorder=2,
            )

        ax.set_title(
            "Neurons Template",
            size=title_size,
        )

        self._set_map_limits(
            ax,
            centers,
            pad=0.7,
            invert_y=True,
        )

        if save:
            path = (
                str(file_path)
                if file_path
                else "Plots/Neurons_template"
            )
            os.makedirs(
                path,
                exist_ok=True,
            )
            fig.savefig(
                os.path.join(
                    path,
                    f"{file_name}_neurons_template.jpg",
                ),
                dpi=300,
                bbox_inches="tight",
            )

        return fig

    # ------------------------------------------------------------------
    # LATTICE GENERATION
    # ------------------------------------------------------------------

    @staticmethod
    def generate_rec_lattice(
        n_columns,
        n_rows,
    ):
        """
        Generate rectangular lattice coordinates.

        Parameters
        ----------
        n_columns : int
            Number of map columns.
        n_rows : int
            Number of map rows.

        Returns
        -------
        ndarray, shape (n_rows * n_columns, 2)
            Coordinates in [x, y] order, flattened row-major.
        """
        n_columns = int(n_columns)
        n_rows = int(n_rows)

        coord_x, coord_y = np.meshgrid(
            np.arange(
                n_columns,
                dtype=float,
            ),
            np.arange(
                n_rows,
                dtype=float,
            ),
            indexing="xy",
        )

        return np.column_stack(
            [
                coord_x.ravel(),
                coord_y.ravel(),
            ]
        )

    @staticmethod
    def generate_hex_lattice(
        n_columns,
        n_rows,
    ):
        """
        Generate odd-r hexagonal lattice coordinates.

        Parameters
        ----------
        n_columns : int
            Number of map columns.
        n_rows : int
            Number of map rows.

        Returns
        -------
        ndarray, shape (n_rows * n_columns, 2)
            Coordinates in [x, y] order, flattened row-major.
        """
        n_columns = int(n_columns)
        n_rows = int(n_rows)

        ratio = (
            np.sqrt(3.0) / 2.0
        )

        coord_x, coord_y = np.meshgrid(
            np.arange(
                n_columns,
                dtype=float,
            ),
            np.arange(
                n_rows,
                dtype=float,
            ),
            indexing="xy",
        )

        coord_y *= ratio
        coord_x[
            1::2,
            :
        ] += 0.5

        return np.column_stack(
            [
                coord_x.ravel(),
                coord_y.ravel(),
            ]
        )

    # ------------------------------------------------------------------
    # HITS
    # ------------------------------------------------------------------

    @property
    def hits_dictionary(self):
        """
        Return hit-size factors indexed by zero-based BMU id.

        The scale remains compatible with the historical visualization:
        0.5 to 2.0.
        """
        unique, counts = np.unique(
            self.bmus,
            return_counts=True,
        )

        if counts.size == 0:
            return {}

        counts = counts.astype(float)

        cmin = float(
            np.min(counts)
        )
        cmax = float(
            np.max(counts)
        )

        if np.isclose(
            cmin,
            cmax,
        ):
            scaled = np.full(
                counts.shape,
                0.5,
                dtype=float,
            )
        else:
            scaled = (
                0.5
                + (
                    counts - cmin
                )
                * (
                    2.0 - 0.5
                )
                / (
                    cmax - cmin
                )
            )

        return dict(
            zip(
                unique.astype(int),
                scaled,
            )
        )

    # ------------------------------------------------------------------
    # TORUS
    # ------------------------------------------------------------------

    def plot_torus(
        self,
        inner_out_prop=0.4,
        red_factor=4,
        hits=False,
        n_colors=32,
        n_training_pixels=5000,
    ):
        """
        Draw the U-Matrix as a torus texture.

        Works for non-square maps and both lattice types because the texture
        is produced from a seamless periodic rendering of the current map.
        """
        if red_factor <= 0:
            raise ValueError(
                "red_factor must be greater than zero."
            )

        mat_im = self.plot_umatrix(
            figsize=(10, 10),
            hits=hits,
            save=False,
            resume=True,
        )

        # Keep the texture orientation stable without tying it to
        # mapsize[0]/mapsize[1]. Rotate BEFORE calculating the target
        # resolution so the texture aspect ratio is preserved.
        if (
            mat_im.shape[0]
            > mat_im.shape[1]
        ):
            mat_im = rotate(
                mat_im,
                90,
                reshape=True,
            )

        y_res = max(
            2,
            int(
                mat_im.shape[0]
                / red_factor
            ),
        )
        x_res = max(
            2,
            int(
                mat_im.shape[1]
                / red_factor
            ),
        )

        mat_res = resize(
            mat_im,
            (
                y_res,
                x_res,
            ),
            preserve_range=True,
            anti_aliasing=True,
        )

        if mat_res.dtype != np.uint8:
            mat_res = np.clip(
                mat_res,
                0,
                255,
            ).astype(
                np.uint8
            )

        def torus(
            rows,
            cols,
            aspect_ratio,
            R_scale=0.4,
        ):
            r_scale = (
                R_scale
                * aspect_ratio
                * inner_out_prop
            )

            u, v = np.meshgrid(
                np.linspace(
                    0,
                    2 * pi,
                    cols,
                    endpoint=False,
                ),
                np.linspace(
                    0,
                    2 * pi,
                    rows,
                    endpoint=False,
                ),
            )

            return (
                (
                    R_scale
                    + r_scale * np.sin(v)
                )
                * np.cos(u),
                (
                    R_scale
                    + r_scale * np.sin(v)
                )
                * np.sin(u),
                r_scale * np.cos(v),
            )

        r, c, _ = mat_res.shape

        aspect_ratio = (
            mat_res.shape[1]
            / mat_res.shape[0]
        )

        x, y, z = torus(
            r,
            c,
            aspect_ratio,
            R_scale=0.4,
        )

        (
            I,
            J,
            K,
            tri_color_intensity,
            pl_colorscale,
        ) = self.mesh_data(
            mat_res,
            n_colors=n_colors,
            n_training_pixels=n_training_pixels,
            periodic=True,
        )

        fig = go.Figure()

        fig.add_mesh3d(
            x=x.ravel(),
            y=y.ravel(),
            z=z.ravel(),
            i=I,
            j=J,
            k=K,
            intensity=tri_color_intensity,
            intensitymode="cell",
            colorscale=pl_colorscale,
            showscale=False,
            flatshading=False,
            lighting=dict(
                ambient=1.0,
                diffuse=0.0,
                specular=0.0,
                roughness=1.0,
                fresnel=0.0,
            ),
            hoverinfo="skip",
        )

        fig.update_layout(
            width=700,
            height=700,
            margin=dict(
                t=10,
                r=10,
                b=10,
                l=10,
            ),
            scene_camera_eye=dict(
                x=-1.75,
                y=-1.75,
                z=1,
            ),
            scene=dict(
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False,
                bgcolor="white",
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            scene_aspectmode="data",
        )

        fig.show()
        return fig

    def mesh_data(
        self,
        img,
        n_colors=32,
        n_training_pixels=800,
        periodic=False,
    ):
        """
        Convert an RGB image to triangle indices and color intensities.
        """
        (
            z_data,
            pl_colorscale,
        ) = self.image2zvals(
            img,
            n_colors=n_colors,
            n_training_pixels=n_training_pixels,
        )

        rows, cols = z_data.shape

        if periodic:
            triangles = self.periodic_tri(
                rows,
                cols,
            )
        else:
            triangles = self.regular_tri(
                rows,
                cols,
            )

        I, J, K = triangles.T

        flattened = z_data.ravel()
        zc = flattened[
            triangles
        ]

        # IMPORTANT:
        # ``z_data`` contains normalized MiniBatchKMeans cluster IDs.
        # These IDs are categorical labels whose numerical order is arbitrary.
        # Averaging them creates colors that do not correspond to any actual
        # texture color and produces striped/mosaic artifacts on the torus.
        #
        # For periodic meshes, always use a REAL palette label from the
        # triangle rather than an arithmetic mean of cluster labels.
        if periodic:
            tri_color_intensity = zc[:, 0]
        else:
            parity = (
                np.arange(
                    zc.shape[0]
                )
                % 2
            )

            tri_color_intensity = np.where(
                parity == 0,
                zc[:, 1],
                zc[:, 2],
            )

        return (
            I,
            J,
            K,
            tri_color_intensity,
            pl_colorscale,
        )

    @staticmethod
    def periodic_tri(
        rows,
        cols,
    ):
        """
        Triangulate a rows x cols grid with periodic wrapping in both axes.

        Unlike :meth:`regular_tri`, this closes the last column onto the first
        and the last row onto the first. This is the correct topology for a
        torus and prevents an open white seam/sector.
        """
        rows = int(rows)
        cols = int(cols)

        if rows < 2 or cols < 2:
            return np.empty(
                (
                    0,
                    3,
                ),
                dtype=int,
            )

        rr, cc = np.meshgrid(
            np.arange(rows, dtype=int),
            np.arange(cols, dtype=int),
            indexing="ij",
        )

        r1 = (
            rr + 1
        ) % rows
        c1 = (
            cc + 1
        ) % cols

        a = (
            rr * cols
            + cc
        ).ravel()
        b = (
            rr * cols
            + c1
        ).ravel()
        d = (
            r1 * cols
            + cc
        ).ravel()
        e = (
            r1 * cols
            + c1
        ).ravel()

        tri1 = np.column_stack(
            [
                a,
                d,
                e,
            ]
        )
        tri2 = np.column_stack(
            [
                a,
                e,
                b,
            ]
        )

        triangles = np.empty(
            (
                2 * rows * cols,
                3,
            ),
            dtype=int,
        )

        triangles[0::2] = tri1
        triangles[1::2] = tri2

        return triangles


    @staticmethod
    def regular_tri(
        rows,
        cols,
    ):
        """
        Vectorized triangulation of a regular rows x cols grid.
        """
        rows = int(rows)
        cols = int(cols)

        if rows < 2 or cols < 2:
            return np.empty(
                (
                    0,
                    3,
                ),
                dtype=int,
            )

        i = np.arange(
            rows - 1,
            dtype=int,
        )[:, None]
        j = np.arange(
            cols - 1,
            dtype=int,
        )[None, :]

        k = (
            j
            + i * cols
        ).ravel()

        tri1 = np.column_stack(
            [
                k,
                k + cols,
                k + cols + 1,
            ]
        )

        tri2 = np.column_stack(
            [
                k,
                k + cols + 1,
                k + 1,
            ]
        )

        triangles = np.empty(
            (
                tri1.shape[0] * 2,
                3,
            ),
            dtype=int,
        )

        triangles[
            0::2
        ] = tri1
        triangles[
            1::2
        ] = tri2

        return triangles

    @staticmethod
    def image2zvals(
        img,
        n_colors=64,
        n_training_pixels=800,
        rngs=123,
    ):
        """
        Quantize image colors using MiniBatchKMeans.

        MiniBatchKMeans is used instead of full KMeans to make torus plotting
        substantially faster for large textures.
        """
        img = np.asarray(img)

        if img.ndim != 3:
            raise ValueError(
                "img must be a color image with shape (rows, cols, channels)."
            )

        rows, cols, channels = img.shape

        if channels < 3:
            raise ValueError(
                "A color image must contain at least 3 channels."
            )

        rgb = img[
            :,
            :,
            :3,
        ].astype(float)

        if (
            np.nanmax(rgb)
            > 1.0
        ):
            rgb /= 255.0

        rgb = np.clip(
            rgb,
            0.0,
            1.0,
        )

        observations = rgb.reshape(
            rows * cols,
            3,
        )

        rng = np.random.default_rng(
            rngs
        )

        n_training = min(
            int(n_training_pixels),
            observations.shape[0],
        )

        if n_training <= 0:
            raise ValueError(
                "n_training_pixels must be greater than zero."
            )

        if (
            n_training
            < observations.shape[0]
        ):
            training_idx = rng.choice(
                observations.shape[0],
                size=n_training,
                replace=False,
            )
            training_pixels = observations[
                training_idx
            ]
        else:
            training_pixels = observations

        n_colors = max(
            1,
            min(
                int(n_colors),
                training_pixels.shape[0],
            ),
        )

        if n_colors == 1:
            color = np.mean(
                training_pixels,
                axis=0,
            )
            z_vals = np.zeros(
                (
                    rows,
                    cols,
                ),
                dtype=float,
            )
            rgb_color = tuple(
                int(v)
                for v in np.clip(
                    color * 255,
                    0,
                    255,
                ).astype(np.uint8)
            )
            color_string = (
                f"rgb({rgb_color[0]}, "
                f"{rgb_color[1]}, "
                f"{rgb_color[2]})"
            )
            return (
                z_vals,
                [
                    [0.0, color_string],
                    [1.0, color_string],
                ],
            )

        model = MiniBatchKMeans(
            n_clusters=n_colors,
            random_state=rngs,
            batch_size=min(
                2048,
                max(
                    256,
                    n_training,
                ),
            ),
            n_init=1,
        )
        model.fit(
            training_pixels
        )

        codebook = model.cluster_centers_
        indices = model.predict(
            observations
        )

        z_vals = (
            indices.astype(float)
            / (
                n_colors - 1
            )
        ).reshape(
            rows,
            cols,
        )

        scale = np.linspace(
            0,
            1,
            n_colors,
        )

        colors = np.clip(
            codebook * 255,
            0,
            255,
        ).astype(
            np.uint8
        )

        pl_colorscale = [
            [
                float(scale_value),
                (
                    f"rgb({int(color[0])}, "
                    f"{int(color[1])}, "
                    f"{int(color[2])})"
                ),
            ]
            for scale_value, color in zip(
                scale,
                colors,
            )
        ]

        return (
            z_vals,
            pl_colorscale,
        )

    # ------------------------------------------------------------------
    # SMALL UTILITIES
    # ------------------------------------------------------------------

    @staticmethod
    def _unique_path(
        filename,
    ):
        """
        Return a non-existing filename by adding _1, _2, ... when needed.
        """
        if not os.path.isfile(
            filename
        ):
            return filename

        name, ext = os.path.splitext(
            filename
        )
        counter = 1

        while os.path.isfile(
            f"{name}_{counter}{ext}"
        ):
            counter += 1

        return (
            f"{name}_{counter}{ext}"
        )

    @staticmethod
    def _component_label_name(
        var_name,
        *,
        full_title=False,
    ):
        """
        Build a filesystem-safe default component filename.
        """
        var_name = str(
            var_name
        )

        if full_title:
            label_name = (
                var_name
                .replace("/", "")
                .replace("\\", "")
            )
        else:
            label_name = (
                var_name[:7]
                .replace(" ", "")
                .replace(":", "")
                .replace("/", "")
                .replace("\\", "")
            )

        return (
            label_name
            if label_name
            else "component"
        )

    def _add_watermark_subplot(
        self,
        fig,
        gs,
        grid_slice=None,
    ):
        """
        Add the IntraSOM watermark if the bundled image is available.
        """
        if self.foot is None:
            return None

        if grid_slice is None:
            grid_slice = (
                slice(95, 100),
                slice(0, 20),
            )

        row_slice, col_slice = grid_slice

        ax = fig.add_subplot(
            gs[
                row_slice,
                col_slice,
            ],
            zorder=-1,
        )
        ax.imshow(
            self.foot,
            aspect="equal",
            alpha=1,
        )
        ax.axis("off")

        return ax
