from math import sqrt
from collections import Counter

import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PolyCollection

from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import davies_bouldin_score as db

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import geopandas as gpd

from importlib import resources
from PIL import Image
from tqdm.notebook import tqdm
from tqdm import tqdm
import plotly.graph_objects as go

class ClusterFactory(object):

    def __init__(self, som_object):
        self.som_object = som_object
        self.name = som_object.name
        self.codebook = som_object.codebook.matrix
        self.mapsize = som_object.mapsize
        self.bmus = som_object._bmu[0].astype(int)
        self.neuron_matrix = som_object.neuron_matrix
        self.component_names = som_object.component_names
        self.unit_names = som_object._unit_names
        self.neurons_dataframe = som_object.neurons_dataframe
        self.sample_names = som_object._sample_names
        self.build_umatrix = som_object.build_umatrix

        # Match visualization.PlotFactory conventions.
        self.mapsize = tuple(int(v) for v in self.mapsize)
        if len(self.mapsize) != 2:
            raise ValueError(
                "mapsize must contain exactly two values: (columns, rows)."
            )

        self.cols, self.rows = self.mapsize
        self.lattice = getattr(som_object, "lattice", "hexa")
        self.mapshape = getattr(som_object, "mapshape", "planar")

        if self.lattice not in {"hexa", "rect"}:
            raise ValueError(
                f"Unsupported lattice {self.lattice!r}. "
                "Accepted values are 'hexa' and 'rect'."
            )

        if self.mapshape not in {"planar", "toroid"}:
            raise ValueError(
                f"Unsupported mapshape {self.mapshape!r}. "
                "Accepted values are 'planar' and 'toroid'."
            )

        if self.codebook.shape[0] != self.cols * self.rows:
            raise ValueError(
                "The codebook number of neurons is inconsistent with mapsize. "
                f"codebook={self.codebook.shape[0]}, "
                f"mapsize={self.mapsize} -> {self.cols * self.rows} neurons."
            )

        self._geometry_cache = {}

        # Load watermark without making plotting fail if the asset is absent.
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
        """Return node coordinates with shape ``(rows, columns)``."""
        cache_key = ("node_coordinates", self.lattice, self.cols, self.rows)

        if cache_key in self._geometry_cache:
            return self._geometry_cache[cache_key]

        if self.lattice == "hexa":
            coordinates = self.generate_hex_lattice(self.cols, self.rows)
        else:
            coordinates = self.generate_rec_lattice(self.cols, self.rows)

        xx = coordinates[:, 0].reshape(self.rows, self.cols)
        yy = coordinates[:, 1].reshape(self.rows, self.cols)

        self._geometry_cache[cache_key] = (xx, yy)
        return xx, yy

    @staticmethod
    def _safe_norm(values):
        """Build a valid Matplotlib normalization, including constant arrays."""
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
    def _polygon_vertices(centers, lattice, sizes=1.0):
        """Generate hexagonal or rectangular cell vertices vectorially."""
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        n = centers.shape[0]

        sizes = np.asarray(sizes, dtype=float)
        if sizes.ndim == 0:
            sizes = np.full(n, float(sizes), dtype=float)
        else:
            sizes = np.broadcast_to(sizes.reshape(-1), (n,)).astype(float)

        if lattice == "hexa":
            angles = (
                np.pi / 2.0
                + np.arange(6, dtype=float) * (2.0 * np.pi / 6.0)
            )
            radius = (1.0 / np.sqrt(3.0)) * sizes
            offsets = np.stack(
                [np.cos(angles), np.sin(angles)],
                axis=1,
            )
            return (
                centers[:, None, :]
                + radius[:, None, None] * offsets[None, :, :]
            )

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
        return (
            centers[:, None, :]
            + half[:, None, None] * base[None, :, :]
        )

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

    def _set_map_limits(self, ax, centers, *, pad=0.75, invert_y=True):
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
        """Use the same flat U-Matrix geometry as ``visualization.PlotFactory``."""
        um = np.asarray(um, dtype=float)
        umat = np.asarray(umat, dtype=float)

        if umat.shape != (self.rows, self.cols):
            raise ValueError(
                "Reduced U-Matrix has an invalid shape. "
                f"Received {umat.shape}, expected {(self.rows, self.cols)}."
            )

        expected_neighbors = 6 if self.lattice == "hexa" else 8
        expected_um = (self.rows, self.cols, expected_neighbors)

        if um.shape != expected_um:
            raise ValueError(
                "Expanded U-Matrix has an invalid shape. "
                f"Received {um.shape}, expected {expected_um}."
            )

        xx, yy = self._node_coordinates()
        node_centers = np.column_stack(
            [(2.0 * xx).ravel(), (2.0 * yy).ravel()]
        )
        node_values = umat.ravel()

        row_grid, col_grid = np.indices(
            (self.rows, self.cols),
            dtype=int,
        )

        if self.lattice == "hexa":
            offsets = np.array(
                [
                    [1.0, 0.0],
                    [0.5, np.sqrt(3.0) / 2.0],
                    [-0.5, np.sqrt(3.0) / 2.0],
                ],
                dtype=float,
            )
            neighbor_slots = (0, 1, 2)
            even_rows = (row_grid % 2) == 0
            neighbor_steps = (
                (np.ones_like(col_grid), np.zeros_like(row_grid)),
                (
                    np.where(even_rows, 0, 1),
                    np.ones_like(row_grid),
                ),
                (
                    np.where(even_rows, -1, 0),
                    np.ones_like(row_grid),
                ),
            )
        else:
            offsets = np.array(
                [
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [-1.0, 1.0],
                ],
                dtype=float,
            )
            neighbor_slots = (0, 1, 2, 3)
            neighbor_steps = (
                (np.ones_like(col_grid), np.zeros_like(row_grid)),
                (np.ones_like(col_grid), np.ones_like(row_grid)),
                (np.zeros_like(col_grid), np.ones_like(row_grid)),
                (-np.ones_like(col_grid), np.ones_like(row_grid)),
            )

        edge_centers = []
        edge_values = []

        for offset, slot, (step_x, step_y) in zip(
            offsets,
            neighbor_slots,
            neighbor_steps,
        ):
            values = np.array(um[:, :, slot], dtype=float, copy=True)

            if self.mapshape == "toroid" and not include_toroid_wrap:
                neighbor_rows = row_grid + step_y
                neighbor_cols = col_grid + step_x
                inside = (
                    (neighbor_rows >= 0)
                    & (neighbor_rows < self.rows)
                    & (neighbor_cols >= 0)
                    & (neighbor_cols < self.cols)
                )
                values[~inside] = np.nan

            edge_centers.append(node_centers + offset)
            edge_values.append(values.ravel())

        return (
            node_centers,
            node_values,
            np.vstack(edge_centers),
            np.concatenate(edge_values),
        )

    def _add_watermark_subplot(
        self,
        fig,
        gs,
        grid_slice=(slice(95, 100), slice(0, 20)),
    ):
        if self.foot is None:
            return None

        ax = fig.add_subplot(gs[grid_slice], zorder=-1)
        ax.imshow(self.foot, aspect="equal", alpha=1)
        ax.axis("off")
        return ax


    def kmeans(self, k=3, init = "random", n_init=5, max_iter=200):
        """
        Runs the K-means algorithm for grouping data from the trained kohonen map.

        Args:
            k (int, optional): The number of desired clusters. The default is 3.
            init (str, optional): Centroid initialization method. Can be 'random' for random startup
                                or 'k-means++' for smart initialization. The default is 'random'.
            n_init (int, optional): Number of times the K-means algorithm will be executed with different initial
                                    centroids. The final result will be the best obtained among the executions.
                                    The default is 5.
            max_iter (int, optional): Maximum number of iterations of the K-means algorithm for each execution.
                                    The default is 200.

        Returns:
            numpy.ndarray: A two-dimensional array containing the cluster labels assigned to each data point.
                        The labels are in the range [1, k]. The form of the array is (self.mapsize[1], self.mapsize[0]).
        """


        kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter).fit(self.codebook).labels_+1

        return kmeans.reshape(self.mapsize[1], self.mapsize[0])

    def results_cluster(self, clusters, save=True, savetype="parquet"):
        """Create the sample-level clustering results DataFrame.

        Each input sample inherits the cluster assigned to its BMU.  This is a
        memory-efficient replacement for the historical implementation: the
        neuron table is indexed directly by the sample BMUs, without first
        copying the complete table and inserting the cluster column into that
        intermediate copy.

        Parameters
        ----------
        clusters : numpy.ndarray
            One- or two-dimensional array containing one cluster label per SOM
            neuron.  The two-dimensional form returned by :meth:`kmeans` is
            accepted directly.
        save : bool, default=True
            Save the resulting table inside the ``Results`` directory.
        savetype : {"parquet", "xlsx", "csv"}, default="parquet"
            Output format used when ``save=True``.

        Returns
        -------
        pandas.DataFrame
            One row per input sample, containing the information of its BMU
            from ``neurons_dataframe`` and the assigned cluster label.
        """
        cluster_array = np.asarray(clusters)
        expected_neurons = self.rows * self.cols

        if cluster_array.ndim not in (1, 2):
            raise ValueError(
                "clusters must be a one- or two-dimensional array. "
                f"Received an array with {cluster_array.ndim} dimensions."
            )

        if cluster_array.size != expected_neurons:
            raise ValueError(
                "clusters must contain exactly one label per SOM neuron. "
                f"Received {cluster_array.size} labels for "
                f"{expected_neurons} neurons."
            )

        if not np.issubdtype(cluster_array.dtype, np.number):
            raise TypeError("clusters must contain numeric labels.")

        flat_clusters = cluster_array.reshape(-1)
        if not np.all(np.isfinite(flat_clusters)):
            raise ValueError("clusters cannot contain NaN or infinite labels.")

        if not np.all(flat_clusters == np.floor(flat_clusters)):
            raise ValueError("clusters must contain integer-valued labels.")

        # int32 is sufficient for cluster labels and halves the memory used by
        # the usual int64 representation in large sample-level result tables.
        flat_clusters = flat_clusters.astype(np.int32, copy=False)

        bmus = np.asarray(self.bmus, dtype=np.intp).reshape(-1)
        if bmus.size == 0:
            raise ValueError("The SOM object does not contain sample BMUs.")

        if np.any((bmus < 0) | (bmus >= expected_neurons)):
            raise ValueError(
                "The SOM object contains BMU indices outside the valid range "
                f"[0, {expected_neurons - 1}]."
            )

        sample_index = pd.Index(self.sample_names, copy=False)
        if len(sample_index) != bmus.size:
            raise ValueError(
                "sample_names and BMUs must have the same length. "
                f"Received {len(sample_index)} names and {bmus.size} BMUs."
            )

        if len(self.neurons_dataframe) != expected_neurons:
            raise ValueError(
                "neurons_dataframe is inconsistent with mapsize. "
                f"Received {len(self.neurons_dataframe)} rows for "
                f"{expected_neurons} neurons."
            )

        max_cluster = int(flat_clusters.max())
        cluster_column = f"{max_cluster}_clusters"

        # Build the final table column by column.  NumPy performs each repeated
        # BMU lookup directly, while ``copy=False`` lets pandas reuse the
        # resulting arrays instead of copying the complete sample table again.
        # The fallback preserves duplicate column names, although the standard
        # IntraSOM neuron table uses unique names.
        neuron_df = self.neurons_dataframe
        if neuron_df.columns.is_unique:
            result_data = {
                column: neuron_df[column].to_numpy(copy=False)[bmus]
                for column in neuron_df.columns
            }
            result_data[cluster_column] = flat_clusters[bmus]
            results_df = pd.DataFrame(
                result_data,
                index=sample_index,
                copy=False,
            )
        else:
            results_df = neuron_df.take(bmus, axis=0)
            results_df[cluster_column] = flat_clusters[bmus]
            results_df.index = sample_index

        if save:
            savetype = str(savetype).lower().lstrip(".")
            path = "Results"
            os.makedirs(path, exist_ok=True)
            file_name = (
                f"{self.name}_results_{max_cluster}_clusters.{savetype}"
            )
            output_path = os.path.join(path, file_name)

            if savetype == "parquet":
                results_df.to_parquet(output_path)
            elif savetype == "xlsx":
                results_df.to_excel(output_path)
            elif savetype == "csv":
                results_df.to_csv(output_path)
            else:
                raise ValueError(
                    "savetype must be 'parquet', 'xlsx', or 'csv'. "
                    f"Received {savetype!r}."
                )

        return results_df
    
    def Davies_Bouldin_analysis(self, 
                                max_clust=30, 
                                n_iter=100, 
                                min_type="ensamble", 
                                plot=True, 
                                save=False,
                                verbose=True):
        """
        DBI vs nº de clusters com tratamento robusto para casos sem 2º/3º mínimos.
        """


        def clust_counter(clusts):
            """
            Conta a frequência dos números de clusters encontrados.

            Sempre retorna array de shape (n, 2):
                coluna 0 = número de clusters
                coluna 1 = percentual

            Se não houver dados válidos:
                shape = (0, 2)
            """

            cl = [
                int(c)
                for c in clusts
                if c is not None
                and not (
                    isinstance(c, (float, np.floating))
                    and np.isnan(c)
                )
            ]

            if len(cl) == 0:
                return np.empty((0, 2), dtype=object)

            count = np.array(
                Counter(cl).most_common(),
                dtype=object
            )

            total = count[:, 1].astype(float).sum()

            if total == 0:
                return np.empty((0, 2), dtype=object)

            count[:, 1] = (
                count[:, 1].astype(float)
                / total
                * 100.0
            )

            return count

        # --------------------------------------------------
        # DATA
        # --------------------------------------------------

        X = np.asarray(
            self.neuron_matrix,
            dtype=float
        )

        n_samples = X.shape[0]

        # Davies-Bouldin requires at least:
        #
        #   2 clusters
        #   and
        #   n_clusters < n_samples
        #
        if n_samples < 3:

            raise ValueError(
                "Davies-Bouldin analysis requires "
                "at least 3 neurons."
            )


        # --------------------------------------------------
        # NORMALIZATION
        # --------------------------------------------------

        mean = np.mean(
            X,
            axis=0
        )

        std = np.std(
            X,
            axis=0
        )

        eps = 1e-12

        std_safe = np.where(
            std < eps,
            1.0,
            std
        )

        X = (
            X - mean
        ) / std_safe


        # --------------------------------------------------
        # VALID CLUSTER RANGE
        # --------------------------------------------------

        max_valid_clust = min(
            int(max_clust),
            n_samples - 1
        )

        if max_valid_clust < 2:

            raise ValueError(
                "The maximum number of clusters "
                "must be at least 2."
            )


        if max_clust > max_valid_clust:

            print(
                f"max_clust={max_clust} is larger than "
                f"the maximum valid value for "
                f"{n_samples} neurons. "
                f"Using max_clust={max_valid_clust}."
            )


        n_clusters = np.arange(
            2,
            max_valid_clust + 1,
            dtype=int
        )
        db_summary = {}

        for it in tqdm(range(n_iter)):
            db_results = np.zeros(len(n_clusters), dtype=float)
            db_summary[f"Iter{it+1}"] = db_results
            for idx, n_clust in enumerate(n_clusters):
                kmeans = KMeans(
                    n_clusters=n_clust,
                    init="random",
                    n_init=5,
                    max_iter=200,
                    random_state=None
                ).fit(X)
                db_results[idx] = db(X, kmeans.labels_)

        df = pd.DataFrame.from_dict(db_summary, orient='index', columns=n_clusters)
        if save:
            path = 'Results'
            os.makedirs(path, exist_ok=True)
            df.to_excel(f"Results/{self.name}_dbindex_iterations.xlsx")

        # --- Cálculo de mínimos (robusto)
        # Nota: local minima com desigualdades estritas; plateaus não contam.
        f_mins, s_mins, t_mins, m_mins, max_mins = [], [], [], [], []
        f_clust_min, s_clust_min, t_clust_min, m_clust_min, max_clust_min = [], [], [], [], []

        for i in range(df.shape[0]):
            it = df.iloc[i].values.astype(float)

            # Índices de mínimos locais
            loc = (np.r_[True, it[1:] < it[:-1]] & np.r_[it[:-1] < it[1:], True])
            local_vals = it[loc]
            local_idxs = np.where(loc)[0]

            # Se não houver mínimos locais, usar mínimo global como "primeiro"
            if local_vals.size == 0:
                # 1º mínimo = mínimo global
                g_idx = int(np.argmin(it))
                g_val = it[g_idx]

                f_val, f_idx = g_val, g_idx
                s_val, s_idx = np.nan, None
                t_val, t_idx = np.nan, None
                m_val, m_idx = np.nan, None
                max_val, max_idx = g_val, g_idx  # por consistência (único disponível)
            else:
                # Ordena mínimos locais por valor ascendente
                order = np.argsort(local_vals)
                sorted_vals = local_vals[order]
                sorted_idxs = local_idxs[order]

                # 1º mínimo
                f_val, f_idx = sorted_vals[0], int(sorted_idxs[0])

                # 2º e 3º se existirem
                if len(sorted_vals) >= 2:
                    s_val, s_idx = sorted_vals[1], int(sorted_idxs[1])
                else:
                    s_val, s_idx = np.nan, None

                if len(sorted_vals) >= 3:
                    t_val, t_idx = sorted_vals[2], int(sorted_idxs[2])
                else:
                    t_val, t_idx = np.nan, None

                # Mediano dos mínimos locais (se houver ≥1)
                mid_pos = len(sorted_vals) // 2
                m_val = sorted_vals[mid_pos] if len(sorted_vals) > 0 else np.nan
                m_idx = int(sorted_idxs[mid_pos]) if len(sorted_vals) > 0 else None

                # Máximo entre os mínimos locais
                max_val = sorted_vals[-1]
                max_idx = int(sorted_idxs[-1])

            # Armazena valores
            f_mins.append(f_val); s_mins.append(s_val); t_mins.append(t_val); m_mins.append(m_val); max_mins.append(max_val)

            # Mapeia para nº de clusters (coluna) — se índice existir
            def idx_to_cluster(idx_):
                if idx_ is None:
                    return None
                # df.columns são os n_clusters (2..max)
                return int(df.columns[idx_])

            f_clust_min.append(idx_to_cluster(f_idx))
            s_clust_min.append(idx_to_cluster(s_idx))
            t_clust_min.append(idx_to_cluster(t_idx))
            m_clust_min.append(idx_to_cluster(m_idx))
            max_clust_min.append(idx_to_cluster(max_idx))

        # Soma para “ensemble” (só os válidos)
        sum_clust_min = [c for c in (f_clust_min + s_clust_min) if c is not None]

        # --- Plot
        if plot:
            x = df.columns.values.astype(int)
            y = df.mean().values
            y_upper = df.quantile(0.75).values
            y_lower = df.quantile(0.25).values
            error = (y_upper - y_lower) / 2

            fig = go.Figure()
            # cuidado com fatia fixa (:,:19) — removido para cobrir todas as iterações
            for i in range(df.shape[0]):
                fig.add_trace(go.Scatter(
                    x=df.columns.values,
                    y=df.iloc[i],
                    name=f"Iter {i+1}",
                    showlegend=False,
                    line=dict(color="silver")
                ))

            # Intervalo interquartil
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                line=dict(color='rgb(0,0,0)'),
                mode='lines',
                name="Trendline",
                error_y=dict(type='data', array=error, visible=True)
            ))

            fig.update_layout(
                template="simple_white",
                title="DBI Comparison: First and Second Global Minima",
                xaxis_title="Cluster Number",
                yaxis_title="Davies-Bouldin Index",
                width=1500,
                height=600,
                font=dict(size=20),
                legend=dict(orientation="h", y=-0.2)
            )

            # Filtra pares válidos para o scatter
            def valid_pairs(xs, ys):
                out_x, out_y = [], []
                for a, b in zip(xs, ys):
                    if a is not None and np.isfinite(b):
                        out_x.append(a); out_y.append(b)
                return out_x, out_y

            vx, vy = valid_pairs(f_clust_min, f_mins)
            fig.add_trace(go.Scatter(x=vx, y=vy, mode='markers', name='Global Minimum',
                                    marker_symbol='circle-open',
                                    marker=dict(size=10, color='#0000FF', line=dict(width=3))))

            vx, vy = valid_pairs(s_clust_min, s_mins)
            fig.add_trace(go.Scatter(x=vx, y=vy, mode='markers', name='Second Global Minimum',
                                    marker_symbol='circle-open',
                                    marker=dict(size=10, color='#DC3912', line=dict(width=3))))
            fig.show()

        # --- Contagens
        fcounter = clust_counter(f_clust_min)
        scounter = clust_counter(s_clust_min)
        tcounter = clust_counter(t_clust_min)
        sumcounter = clust_counter(sum_clust_min)


        # ---------------------------------------------------------
        # Funções auxiliares seguras
        # ---------------------------------------------------------

        def counter_value(counter, position=0):
            """
            Retorna o número de clusters de uma posição do contador.

            Se essa posição não existir, retorna None.
            """

            if counter.shape[0] <= position:
                return None

            return int(counter[position, 0])


        def counter_percentage(counter, position=0):
            """
            Retorna o percentual de uma posição do contador.

            Se essa posição não existir, retorna None.
            """

            if counter.shape[0] <= position:
                return None

            return float(counter[position, 1])


        def print_counter(counter, title, max_items=2):

            print(f"{title}:")

            if counter.shape[0] == 0:
                print("Sem mínimo disponível.")
                return

            for i in range(min(max_items, counter.shape[0])):

                cluster = int(counter[i, 0])
                percentage = float(counter[i, 1])

                print(
                    f"Cluster Number: {cluster}. "
                    f"Percentage: {percentage:.2f}%."
                )


        # ---------------------------------------------------------
        # Valores principais
        # ---------------------------------------------------------

        first_minimum = counter_value(fcounter, 0)
        second_minimum = counter_value(scounter, 0)
        third_minimum = counter_value(tcounter, 0)
        ensemble_minimum = counter_value(sumcounter, 0)


        # ---------------------------------------------------------
        # Saída
        # ---------------------------------------------------------

        if verbose:

            print("Davies-Bouldin results:\n")

            print_counter(
                fcounter,
                "First Minimum"
            )

            print()

            print_counter(
                scounter,
                "Second Minimum"
            )

            print()

            print_counter(
                tcounter,
                "Third Minimum"
            )

            print()

            print_counter(
                sumcounter,
                "Ensemble Minimum"
            )

        else:

            print("Davies-Bouldin results:")

            print(
                f"First Minimum: "
                f"{first_minimum if first_minimum is not None else 'N/A'}"
            )

            print(
                f"Second Minimum: "
                f"{second_minimum if second_minimum is not None else 'N/A'}"
            )

            print(
                f"Third Minimum: "
                f"{third_minimum if third_minimum is not None else 'N/A'}"
            )

            print(
                f"Ensemble Minimum: "
                f"{ensemble_minimum if ensemble_minimum is not None else 'N/A'}"
            )


        # ---------------------------------------------------------
        # Escolhe resultado conforme min_type
        # ---------------------------------------------------------

        min_type_lower = str(min_type).lower()

        if min_type_lower in ("first", "first_minimum"):

            selected = first_minimum

        elif min_type_lower in ("second", "second_minimum"):

            # Se não existe segundo mínimo, usa o primeiro
            selected = (
                second_minimum
                if second_minimum is not None
                else first_minimum
            )

        elif min_type_lower in (
            "ensamble",
            "ensemble",
            "ensemble_minimum"
        ):

            selected = (
                ensemble_minimum
                if ensemble_minimum is not None
                else first_minimum
            )

        else:

            raise ValueError(
                "min_type deve ser 'first', 'second' "
                "ou 'ensamble'."
            )


        # Última proteção
        if selected is None:
            raise RuntimeError(
                "Não foi possível determinar um número "
                "válido de clusters pelo índice Davies-Bouldin."
            )


        return int(selected)

            
    def plot_kmeans(
        self,
        clusters,
        figsize=(16, 14),
        title_size=25,
        title_pad=40,
        legend_title=False,
        legend_text_size=10,
        save=False,
        file_name=None,
        file_path=False,
        watermark_neurons=False,
        neurons_fontsize=12,
        umatrix=False,
        hits=False,
        alfa_clust=0.5,
        log=False,
        colormap="gist_rainbow",
        clusters_highlight=None,
        legend_title_size=12,
        cluster_outline=False,
        plot_labels=False,
        custom_labels=None,
        clusterout_maxtext_size=12,
        return_geodataframe=False,
        auto_adjust_text=False,
    ):
        """
        Plot K-means clusters using the same map geometry as ``PlotFactory``.

        All historical visualization features are retained:
        U-Matrix background, hits, neuron watermark, highlighted clusters,
        dissolved/fused cluster polygons, labels, custom labels, legend,
        saving and optional GeoDataFrame return.
        """
        clusters = np.asarray(clusters)
        expected_shape = (self.rows, self.cols)

        if clusters.shape != expected_shape:
            if clusters.size == self.rows * self.cols:
                clusters = clusters.reshape(expected_shape)
            else:
                raise ValueError(
                    "clusters has an invalid shape. "
                    f"Received {clusters.shape}, expected {expected_shape}."
                )

        if not np.issubdtype(clusters.dtype, np.number):
            raise TypeError("clusters must contain numeric labels.")

        if not np.all(np.isfinite(clusters)):
            raise ValueError("clusters cannot contain NaN or infinite labels.")

        clusters_highlight = (
            []
            if clusters_highlight is None
            else list(clusters_highlight)
        )
        custom_labels = (
            []
            if custom_labels is None
            else list(custom_labels)
        )

        unique_labels = np.sort(np.unique(clusters))
        n_clusters = len(unique_labels)

        if file_name is None:
            file_name = f"Clusters_{n_clusters}_{self.name}"

        fig = plt.figure(figsize=figsize, dpi=300)
        gs = gridspec.GridSpec(100, 100, figure=fig)

        n_legend_items = (
            n_clusters
            if not clusters_highlight
            else len(clusters_highlight)
        )
        n_legend_items += int(watermark_neurons)

        pad_subplots = 3
        if n_legend_items <= 10:
            map_end = 90 - pad_subplots
            legend_start = 90
        elif n_legend_items <= 20:
            map_end = 80 - pad_subplots
            legend_start = 80
        elif n_legend_items <= 30:
            map_end = 70 - pad_subplots
            legend_start = 70
        else:
            map_end = 60 - pad_subplots
            legend_start = 60

        ax = fig.add_subplot(gs[:95, :map_end])
        ax.set_aspect("equal")

        xx, yy = self._node_coordinates()
        node_centers = np.column_stack(
            [(2.0 * xx).ravel(), (2.0 * yy).ravel()]
        )

        # --------------------------------------------------------------
        # Optional U-Matrix background and hits
        # --------------------------------------------------------------
        all_centers_for_limits = [node_centers]

        if umatrix:
            um = np.asarray(
                self.build_umatrix(expanded=True, log=log),
                dtype=float,
            )
            umat = np.asarray(
                self.build_umatrix(expanded=False, log=log),
                dtype=float,
            )

            (
                umat_node_centers,
                umat_node_values,
                edge_centers,
                edge_values,
            ) = self._umatrix_geometry(
                um,
                umat,
                include_toroid_wrap=False,
            )

            finite_values = np.concatenate(
                [
                    umat_node_values[np.isfinite(umat_node_values)],
                    edge_values[np.isfinite(edge_values)],
                ]
            )
            umat_norm = self._safe_norm(finite_values)
            umat_cmap = mpl.colormaps["jet"]

            self._add_value_cells(
                ax,
                umat_node_centers,
                umat_node_values,
                cmap=umat_cmap,
                norm=umat_norm,
                zorder=0,
            )
            self._add_value_cells(
                ax,
                edge_centers,
                edge_values,
                cmap=umat_cmap,
                norm=umat_norm,
                zorder=0,
            )

            finite_edge_centers = edge_centers[np.isfinite(edge_values)]
            if finite_edge_centers.size:
                all_centers_for_limits.append(finite_edge_centers)

            if hits:
                hit_sizes_dict = self.hits_dictionary
                hit_nodes = np.array(
                    sorted(hit_sizes_dict.keys()),
                    dtype=int,
                )
                valid = (
                    (hit_nodes >= 0)
                    & (hit_nodes < node_centers.shape[0])
                )
                hit_nodes = hit_nodes[valid]

                if hit_nodes.size:
                    self._add_solid_cells(
                        ax,
                        node_centers[hit_nodes],
                        facecolors="white",
                        edgecolors="lightgray",
                        linewidths=0.4,
                        sizes=np.array(
                            [
                                hit_sizes_dict[int(node)]
                                for node in hit_nodes
                            ],
                            dtype=float,
                        ),
                        zorder=0.5,
                    )

        # --------------------------------------------------------------
        # Cluster colors
        # --------------------------------------------------------------
        cluster_norm = self._safe_norm(unique_labels)
        cmap = mpl.colormaps.get_cmap(colormap)

        def cluster_color(label):
            if clusters_highlight and label not in clusters_highlight:
                return "gray"
            return cmap(cluster_norm(float(label)))

        flat_labels = clusters.ravel()
        facecolors = np.array(
            [cluster_color(label) for label in flat_labels],
            dtype=object,
        )

        gdf = None

        # --------------------------------------------------------------
        # Fused/dissolved cluster polygons
        # --------------------------------------------------------------
        if cluster_outline:
            # The historical plot used radius=2/sqrt(3)+0.04. Convert that
            # exactly to the common ``sizes`` convention.
            outline_size = 2.0 + 0.04 * np.sqrt(3.0)
            vertices = self._polygon_vertices(
                node_centers,
                self.lattice,
                sizes=outline_size,
            )

            polygons_by_label = {
                label: []
                for label in unique_labels
            }

            for label, verts in zip(flat_labels, vertices):
                polygons_by_label[label].append(Polygon(verts))

            records = []
            for position, label in enumerate(unique_labels):
                dissolved = unary_union(polygons_by_label[label])

                # Preserve the original inward buffer used to create a clean,
                # visible separation between neighboring fused groups.
                dissolved = dissolved.buffer(-0.075)

                if dissolved.is_empty:
                    dissolved = unary_union(polygons_by_label[label])

                if custom_labels:
                    if len(custom_labels) != n_clusters:
                        raise ValueError(
                            "custom_labels must contain one label for each "
                            f"cluster ({n_clusters})."
                        )
                    display_label = custom_labels[position]
                else:
                    display_label = f"#{label:g}"

                records.append(
                    {
                        "geometry": dissolved,
                        "color": cluster_color(label),
                        "label": display_label,
                        "cluster": label,
                    }
                )

            gdf = gpd.GeoDataFrame(records, geometry="geometry")
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)

            gdf.plot(
                ax=ax,
                facecolor=gdf["color"],
                edgecolor="none",
                alpha=alfa_clust,
                zorder=1,
            )
            gdf.plot(
                ax=ax,
                facecolor="none",
                edgecolor=gdf["color"],
                alpha=1,
                linewidth=2,
                zorder=1.1,
            )

            if plot_labels:
                for _, row in gdf.iterrows():
                    if (
                        clusters_highlight
                        and row["cluster"] not in clusters_highlight
                    ):
                        continue

                    polygon = row.geometry
                    label = str(row["label"])

                    if polygon.is_empty:
                        continue

                    if auto_adjust_text:
                        rectangle = polygon.minimum_rotated_rectangle
                        coords = list(rectangle.exterior.coords)

                        edge_lengths = [
                            Point(coords[i]).distance(Point(coords[i + 1]))
                            for i in range(4)
                        ]
                        major_index = int(np.argmax(edge_lengths))
                        p0 = coords[major_index]
                        p1 = coords[major_index + 1]
                        angle = np.degrees(
                            np.arctan2(
                                p1[1] - p0[1],
                                p1[0] - p0[0],
                            )
                        )
                        if angle > 90:
                            angle -= 180
                        elif angle < -90:
                            angle += 180

                        point = polygon.representative_point()
                        minx, miny, maxx, maxy = rectangle.bounds
                        available = max(
                            min(maxx - minx, maxy - miny),
                            1e-6,
                        )
                        estimated_text_width = max(
                            len(label) * 0.6,
                            1.0,
                        )
                        fontsize = min(
                            clusterout_maxtext_size,
                            max(
                                4.0,
                                clusterout_maxtext_size
                                * available
                                / estimated_text_width,
                            ),
                        )

                        ax.text(
                            point.x,
                            point.y,
                            label,
                            ha="center",
                            va="center",
                            fontsize=fontsize,
                            rotation=angle,
                            color="white",
                            weight="bold",
                            zorder=3,
                        )
                    else:
                        point = polygon.representative_point()
                        ax.text(
                            point.x + 0.05,
                            point.y + 0.05,
                            label,
                            ha="center",
                            va="center",
                            color="black",
                            alpha=0.7,
                            weight="bold",
                            fontsize=clusterout_maxtext_size,
                            zorder=3,
                        )
                        ax.text(
                            point.x,
                            point.y,
                            label,
                            ha="center",
                            va="center",
                            color="white",
                            weight="bold",
                            fontsize=clusterout_maxtext_size,
                            zorder=3.1,
                        )

        # --------------------------------------------------------------
        # Individual cluster cells
        # --------------------------------------------------------------
        else:
            cluster_size = 2.0 * 0.96
            self._add_solid_cells(
                ax,
                node_centers,
                facecolors=facecolors,
                edgecolors=facecolors,
                linewidths=1.9,
                alpha=alfa_clust,
                sizes=cluster_size,
                zorder=1,
            )

        # --------------------------------------------------------------
        # Neuron watermark
        # --------------------------------------------------------------
        if watermark_neurons:
            self._add_solid_cells(
                ax,
                node_centers,
                facecolors="white",
                edgecolors="black",
                linewidths=0.6,
                alpha=0.1,
                sizes=2.0,
                zorder=2,
            )

            for node_id, (x, y) in enumerate(node_centers, start=1):
                ax.text(
                    x,
                    y,
                    str(node_id),
                    size=neurons_fontsize,
                    ha="center",
                    va="center",
                    color="black",
                    zorder=2.1,
                )

        all_centers = np.vstack(all_centers_for_limits)
        map_pad = 1.1
        if watermark_neurons:
            map_pad = max(map_pad, 1.4)

        self._set_map_limits(
            ax,
            all_centers,
            pad=map_pad,
            invert_y=True,
        )

        ax.set_title(
            f"Clustering Matrix - {n_clusters} clusters",
            ha="center",
            va="top",
            size=title_size,
            pad=title_pad,
        )

        # --------------------------------------------------------------
        # Legend
        # --------------------------------------------------------------
        ax2 = fig.add_subplot(gs[20:80, legend_start:])
        ax2.invert_yaxis()
        ax2.set_aspect("equal")

        legend_labels = (
            unique_labels.tolist()
            if not clusters_highlight
            else [
                label
                for label in clusters_highlight
                if label in set(unique_labels.tolist())
            ]
        )

        n_items = len(legend_labels) + int(watermark_neurons)
        n_cols = max(1, int(np.ceil(n_items / 10)))
        n_rows = max(1, int(np.ceil(n_items / n_cols)))

        cell_height = 0.096
        pad = 0.1 - cell_height
        radius = cell_height / 2
        total_height = cell_height * n_rows + n_rows * pad
        shift = cell_height
        y_start = ((1 - total_height) / 2) + shift / 2
        x_start = pad + shift / 2
        text_pad = cell_height * 3

        legend_entries = [
            ("cluster", label)
            for label in legend_labels
        ]
        if watermark_neurons:
            legend_entries.append(("neuron", None))

        label_to_position = {
            label: pos
            for pos, label in enumerate(unique_labels)
        }

        for i, (entry_type, label) in enumerate(legend_entries):
            xfac = i // n_rows
            yfac = i % n_rows

            x_center = (
                x_start
                + xfac * shift
                + xfac * pad
                + xfac * text_pad
            )
            y_center = y_start + yfac * shift + yfac * pad

            if entry_type == "cluster":
                color = cluster_color(label)
                marker = RegularPolygon(
                    (x_center, y_center),
                    numVertices=6 if self.lattice == "hexa" else 4,
                    radius=radius,
                    orientation=0 if self.lattice == "hexa" else np.pi / 4,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=2,
                    alpha=alfa_clust,
                )
                ax2.add_patch(marker)

                if custom_labels:
                    cluster_name = custom_labels[
                        label_to_position[label]
                    ]
                else:
                    cluster_name = f"Cluster #{label:g}"

                ax2.annotate(
                    cluster_name,
                    xy=(x_center + radius + 0.01, y_center),
                    xytext=(0, 0),
                    textcoords="offset points",
                    color="black",
                    weight="bold",
                    fontsize=legend_text_size,
                    ha="left",
                    va="center",
                )
            else:
                marker = RegularPolygon(
                    (x_center, y_center),
                    numVertices=6 if self.lattice == "hexa" else 4,
                    radius=radius * 0.95,
                    orientation=0 if self.lattice == "hexa" else np.pi / 4,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=2,
                )
                ax2.add_patch(marker)
                ax2.annotate(
                    "#",
                    xy=(x_center, y_center),
                    xytext=(0, 0),
                    textcoords="offset points",
                    color="black",
                    weight="bold",
                    fontsize=legend_text_size,
                    ha="center",
                    va="center",
                )
                ax2.annotate(
                    "Neuron Number",
                    xy=(x_center + radius + 0.01, y_center),
                    xytext=(0, 0),
                    textcoords="offset points",
                    color="black",
                    weight="bold",
                    fontsize=legend_text_size,
                    ha="left",
                    va="center",
                )

        ax2.set_title(
            legend_title if legend_title is not False else "Legend",
            fontdict={"fontsize": legend_title_size},
            loc="center",
            pad=5,
            fontweight="bold",
            y=1 - y_start + 0.03,
        )
        ax2.set_xlim(
            0,
            n_cols * cell_height + n_cols * pad + n_cols * text_pad,
        )
        ax2.set_ylim(1, -0.01)
        ax2.set_axis_off()

        self._add_watermark_subplot(fig, gs)
        fig.subplots_adjust(wspace=0.1)

        if save:
            path = (
                str(file_path)
                if file_path
                else "Plots/Clusters"
            )
            os.makedirs(path, exist_ok=True)
            fig.savefig(
                os.path.join(path, f"{file_name}.jpg"),
                dpi=300,
                bbox_inches="tight",
            )

        if return_geodataframe:
            if not cluster_outline:
                raise ValueError(
                    "return_geodataframe=True requires "
                    "cluster_outline=True."
                )
            return gdf

        return fig


    @staticmethod
    def generate_rec_lattice(n_columns, n_rows):
        """Generate rectangular coordinates in row-major order."""
        n_columns = int(n_columns)
        n_rows = int(n_rows)

        coord_x, coord_y = np.meshgrid(
            np.arange(n_columns, dtype=float),
            np.arange(n_rows, dtype=float),
            indexing="xy",
        )
        return np.column_stack(
            [coord_x.ravel(), coord_y.ravel()]
        )

    @staticmethod
    def generate_hex_lattice(n_columns, n_rows):
        """Generate odd-r hexagonal coordinates in row-major order."""
        n_columns = int(n_columns)
        n_rows = int(n_rows)
        ratio = np.sqrt(3.0) / 2.0

        coord_x, coord_y = np.meshgrid(
            np.arange(n_columns, dtype=float),
            np.arange(n_rows, dtype=float),
            indexing="xy",
        )
        coord_y *= ratio
        coord_x[1::2, :] += 0.5

        return np.column_stack(
            [coord_x.ravel(), coord_y.ravel()]
        )


    # def build_umatrix(self, expanded=False, log=False):
    #     """
    #     Function to calculate the U Matrix of unified distances from the
    #     trained weight matrix.
    # 
    #     Args:
    #         expanded: boolean value to indicate whether the return will be from the summarized
    #             or unified matrix of distances (average of distances from the 6
    #             neighborhood BMUs) or expanded (all distance values)
    #             
    #     Returns:
    #         Expanded or summarized unified distance matrix.
    #     """
    #     # Function to find distance quickly
    #     def fast_norm(x):
    #         """
    #         Returns the L2 norm of a 1-D array.
    #         """
    #         return sqrt(dot(x, x.T))
    # 
    #     # Matrix of BMUs weights
    #     weights = np.reshape(self.codebook, (self.mapsize[1], self.mapsize[0], self.codebook.shape[1]))
    # 
    #     # Neighbor hexagonal search
    #     ii = [[1, 1, 0, -1, 0, 1], [1, 0,-1, -1, -1, 0]]
    #     jj = [[0, 1, 1, 0, -1, -1], [0, 1, 1, 0, -1, -1]]
    # 
    #     # Initialize U Matrix
    #     um = np.nan * np.zeros((weights.shape[0], weights.shape[1], 6))
    # 
    #     # Fill U Matrix
    #     for y in range(weights.shape[0]):
    #         for x in range(weights.shape[1]):
    #             w_2 = weights[y, x]
    #             e = y % 2 == 0
    #             for k, (i, j) in enumerate(zip(ii[e], jj[e])):
    #                 if (x+i >= 0 and x+i < weights.shape[1] and y+j >= 0 and y+j < weights.shape[0]):
    #                     w_1 = weights[y+j, x+i]
    #                     um[y, x, k] = fast_norm(w_2-w_1)
    #     if expanded:
    #         # Expanded U matrix
    #         return np.log(um) if log else um
    #     else:
    #         # Reduced U matrix
    #         return nanmean(np.log(um), axis=2) if log else nanmean(um, axis=2)
        

    @property
    def hits_dictionary(self):
        """
        Function to create a dictionary of hits from the input vectors for
        each of its BMUs, proportional to the size of the plot.
        """
        # Hit count
        unique, counts = np.unique(self.bmus, return_counts=True)

        # Normalize this count from 0.5 to 2.0 (from a small hexagon to a
        # hexagon that covers half of the neighbors).
        counts = minmax_scale(counts, feature_range = (0.5,2))

        return dict(zip(unique, counts))

# K-Means error
os.environ["OMP_NUM_THREADS"] = "2"
