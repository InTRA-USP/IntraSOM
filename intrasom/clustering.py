from math import sqrt
from collections import Counter

import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import RegularPolygon

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
        # Load foot image
        image_file = resources.files("intrasom") / "images" / "foot.jpg"
        self.foot = Image.open(image_file)

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

        # --- Dados
        X = np.asarray(self.neuron_matrix)

        # Normalização robusta (evita std==0)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        eps = 1e-12
        std_safe = np.where(std < eps, 1.0, std)
        X = (X - mean) / std_safe

        # --- Varredura DBI
        n_clusters = np.arange(2, max_clust + 1, 1)
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

            
    def plot_kmeans(self, 
                    clusters, 
                    figsize = (16,14),
                    title_size = 25,
                    title_pad = 40,
                    legend_title = False,
                    legend_text_size = 10,
                    save = False,
                    file_name = None,
                    file_path = False,
                    watermark_neurons = False, 
                    neurons_fontsize = 12,
                    umatrix = False, 
                    hits=False, 
                    alfa_clust = 0.5, 
                    log=False,
                    colormap = "gist_rainbow",
                    clusters_highlight = [],
                    legend_title_size=12,
                    cluster_outline=False,
                    plot_labels=False,
                    custom_labels = [],
                    clusterout_maxtext_size=12,
                    return_geodataframe=False,
                    auto_adjust_text =False):
        """
        Plots a graph with the clusters resulting from the execution of the K-means algorithm.

        Parameters:
        - clusters: ndarray or 2-dimensional list
            Array or 2-dimensional list containing the clusters generated by the K-means algorithm.
        - figsize: tuple, optional
            Figure size (width, height). Default is (16, 14).
        - title_size: int, optional
            Font size of the title. Default is 25.
        - title_pad: int, optional
            Spacing between the title and the top of the graph. Default is 40.
        - legend_text_size: int, optional
            Font size of the legend text. Default is 10.
        - save: bool, optional
            Indicates whether the graph should be saved to a file. Default is False.
        - file_name: str, optional
            Name of the file to be saved. Default is None.
        - file_path: bool, optional
            Indicates whether the file path should be included when saving. Default is False.
        - watermark_neurons: bool, optional
            Indicates whether the BMU numbers should be displayed as a watermark on the graph. Default is False.
        - neurons_fontsize: int, optional
            Font size of the BMU numbers. Default is 12.
        - umatrix: bool, optional
            Indicates whether the U-matrix should be plotted on the graph. Default is False.
        - hits: bool, optional
            Indicates whether the hits should be plotted on the graph. Default is False.
        - alfa_clust: float, optional
            Transparency value of the clusters in the graph. Default is 0.5.
        - log: bool, optional
            Indicates whether the U-matrix should be plotted on a logarithmic scale. Default is False.
        - colormap: str, optional
            Name of the colormap to be used in the graph. Default is "gist_rainbow".
        - clusters_highlight: list, optional
            List containing the clusters that should be highlighted in the graph. Default is [].
        - legend_title_size: int, optional
            Font size of the legend title. Default is 12.
        - cluster_outline: bool, optional
            Indicates whether the cluster outlines should be drawn on the graph. Default is False.
        - plot_labels: bool, optional
            Indicates whether the labels should be plotted on the graph. Default is False.
        - custom_labels: list, optional
            List containing custom labels for each point on the graph. Default is [].
        - clusterout_maxtext_size: int, optional
            Maximum size of the cluster text on the graph. Default is 12.
        - return_geodataframe: bool, optional
            Indicates whether the GeoDataFrame should be returned. Default is False.
        - auto_adjust_text: bool, optional
            Indicates whether automatic text adjustment should be enabled. Default is False.
        """
        

        if file_name is None:
            file_name = f"Clusters_{len(np.unique(clusters))}_{self.name}"


        f = plt.figure(figsize=figsize, dpi=300)
        gs = gridspec.GridSpec(100, 100)
        
        max_clust = clusters.max() if len(clusters_highlight) == 0 else len(clusters_highlight)
        max_clust = max_clust+1 if watermark_neurons else max_clust

        pad_subplots = 3
        if max_clust <=10:
            ax = f.add_subplot(gs[:95, :90-pad_subplots])
        elif max_clust<=20:
            ax = f.add_subplot(gs[:95, :80-pad_subplots])
        elif max_clust<=30:
            ax = f.add_subplot(gs[:95, :70-pad_subplots])
        else:
            ax = f.add_subplot(gs[:95, :60-pad_subplots])

        ax.set_aspect('equal')

        xx = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,0], (self.mapsize[1], self.mapsize[0]))
        yy = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,1], (self.mapsize[1], self.mapsize[0]))

        if umatrix:
            if hits:
                bmu_dic = self.hits_dictionary
            # U Matrix
            um = self.build_umatrix(expanded = True, log=log)
            umat = self.build_umatrix(expanded = False, log=log)

            # Normalize the colors for all hexagons
            norm = mpl.colors.Normalize(vmin=np.nanmin(um), vmax=np.nanmax(um))
            counter = 0

            alfa = 1

            for j in range(self.mapsize[1]):
                for i in range(self.mapsize[0]):
                    # Central Hexagon
                    hexagon = RegularPolygon((xx[(j, i)]*2,
                                          yy[(j,i)]*2),
                                         numVertices=6,
                                         radius=1/np.sqrt(3),
                                         facecolor= mpl.colormaps["jet"](norm(umat[j][i])),
                                         alpha=alfa, 
                                         zorder=0)#, edgecolor='black')

                    ax.add_patch(hexagon)

                    # Right Hexagon
                    if not np.isnan(um[j, i, 0]):
                        hexagon = RegularPolygon((xx[(j, i)]*2+1,
                                              yy[(j,i)]*2),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=mpl.colormaps["jet"](norm(um[j,i,0])),
                                             alpha=alfa, 
                                             zorder=0)
                        ax.add_patch(hexagon)

                    # Upper Right Hexagon
                    if not np.isnan(um[j, i, 1]):
                        hexagon = RegularPolygon((xx[(j, i)]*2+0.5,
                                              yy[(j,i)]*2+(np.sqrt(3)/2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=mpl.colormaps["jet"](norm(um[j,i,1])),
                                             alpha=alfa, 
                                             zorder=0)
                        ax.add_patch(hexagon)

                    # Upper Left Hexagon
                    if not np.isnan(um[j, i, 2]):
                        hexagon = RegularPolygon((xx[(j, i)]*2-0.5,
                                              yy[(j,i)]*2+(np.sqrt(3)/2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=mpl.colormaps["jet"](norm(um[j,i,2])),
                                             alpha=alfa, 
                                             zorder=0)
                        ax.add_patch(hexagon)

                    # Plot hits
                    if hits:
                        try:
                            hexagon = RegularPolygon((xx[(j, i)]*2,
                                                  yy[(j,i)]*2),
                                                 numVertices=6,
                                                 radius=((1/np.sqrt(3))*bmu_dic[counter]),
                                                 facecolor='white',
                                                 alpha=alfa, 
                                                 zorder=0)
                            ax.add_patch(hexagon)
                        except:
                            pass
                    counter+=1
        
        norm = mpl.colors.Normalize(vmin=np.nanmin(clusters), vmax=np.nanmax(clusters))

        cmap = mpl.colormaps.get_cmap(colormap)
        if cluster_outline:
            cluster_vertices_dict = {}

            for j in range(clusters.shape[0]):
                for i in range(clusters.shape[1]):

                    label = clusters[j][i]

                    if len(clusters_highlight) == 0:
                        color = cmap(norm(clusters[j][i]))
                    else:   
                        color = "gray" if clusters[j][i] not in clusters_highlight else cmap(norm(clusters[j][i]))

                    if label not in cluster_vertices_dict:
                        cluster_vertices_dict[label] = []
                    
                    # Get the vertices of the hexagon
                    hexagon = RegularPolygon((xx[(j, i)]*2, yy[(j,i)]*2), 
                                        numVertices=6, 
                                        radius=2/np.sqrt(3)+0.04,
                                        facecolor = color, 
                                        alpha = alfa_clust)
                    vertices = hexagon.get_verts()
                    polygon = Polygon(vertices)
                    cluster_vertices_dict[label].append(polygon)

                    if watermark_neurons:
                        # For use in BMUs number plot
                        nnodes = self.mapsize[0] * self.mapsize[1]
                        grid_bmus = np.linspace(1,nnodes, nnodes).reshape(self.mapsize[1], self.mapsize[0])
                        # Central Hexagon
                        hexagon = RegularPolygon((xx[(j, i)]*2,
                                                yy[(j,i)]*2),
                                                numVertices=6,
                                                radius=2/np.sqrt(3),
                                                facecolor= "white",
                                                edgecolor='black',
                                                alpha=0.1, 
                                                zorder=2)#, edgecolor='black')
                        ax.add_patch(hexagon)

                        ax.text(xx[(j,i)]*2, yy[(j,i)]*2, 
                                s=f"{int(grid_bmus[j,i])}", 
                                size = neurons_fontsize,
                                horizontalalignment='center', 
                                verticalalignment='center', 
                                color='black', 
                                zorder=2)
        else:
            for j in range(clusters.shape[0]):
                for i in range(clusters.shape[1]):

                    if len(clusters_highlight) == 0:
                        color = cmap(norm(clusters[j][i]))
                    else:
                        color = "gray" if clusters[j][i] not in clusters_highlight else cmap(norm(clusters[j][i]))

                    hexagon = RegularPolygon((xx[(j, i)]*2, yy[(j,i)]*2), 
                                        numVertices=6, 
                                        radius=2/np.sqrt(3)-0.04*(2/np.sqrt(3)),
                                        facecolor = color, 
                                        alpha = alfa_clust, 
                                         zorder=1)
                    ax.add_patch(hexagon)                
                    
                    hexagon = RegularPolygon((xx[(j, i)]*2, yy[(j,i)]*2), 
                                        numVertices=6, 
                                        radius=2/np.sqrt(3)-0.04*(2/np.sqrt(3)),
                                        fill=False,
                                        facecolor = None, 
                                        edgecolor = color,
                                        linewidth=1.9, 
                                        alpha = 1, 
                                        zorder = 1)
                    ax.add_patch(hexagon)
                
                    if watermark_neurons:
                        # For use in neurons number plot
                        nnodes = self.mapsize[0] * self.mapsize[1]
                        grid_bmus = np.linspace(1,nnodes, nnodes).reshape(self.mapsize[1], self.mapsize[0])
                        # Central Hexagon
                        hexagon = RegularPolygon((xx[(j, i)]*2,
                                                yy[(j,i)]*2),
                                                numVertices=6,
                                                radius=2/np.sqrt(3),
                                                facecolor= "white",
                                                edgecolor='black',
                                                alpha=0.1, 
                                                zorder=2)#, edgecolor='black')
                        ax.add_patch(hexagon)

                        ax.text(xx[(j,i)]*2, yy[(j,i)]*2, 
                                s=f"{int(grid_bmus[j,i])}", 
                                size = neurons_fontsize,
                                horizontalalignment='center', 
                                verticalalignment='center', 
                                color='black', 
                                zorder=2)

        if cluster_outline:
            cluster_vertices_dict = dict(sorted(cluster_vertices_dict.items()))

            # Create a list of GeoSeries for each cluster
            cluster_geo_series = []
            for label in cluster_vertices_dict:
                cluster_geo_series.append(gpd.GeoSeries(cluster_vertices_dict[label]))

            # Dissolve the polygons in each GeoSeries
            dissolved_geometries = []
            for geo_series in cluster_geo_series:
                dissolved_geometry = unary_union(geo_series)
                dissolved_geometries.append(dissolved_geometry)

            # Colors of the clusters
            colors = []
            labels_default = []
            cluster = []
            for i in range(len(dissolved_geometries)):
                label = i+1
                if len(clusters_highlight) == 0:
                    color = cmap(norm(label))
                else:
                    color = "gray" if label not in clusters_highlight else cmap(norm(label))
                
                colors.append(color)
                labels_default.append(f"#{label}")
                cluster.append(label)

            # Create a GeoDataFrame from the list of geometries
            gdf = gpd.GeoDataFrame(geometry=[geom for geom in dissolved_geometries])

            # Buffer inside to plot the outline
            gdf["geometry"] = gdf.buffer(-0.075)

            # Add a new column to the GeoDataFrame with the colors
            gdf['color'] = colors
            gdf['label'] = labels_default
            gdf['cluster'] = cluster

            if len(custom_labels)>0:
                gdf['label'] = custom_labels

            gdf = gdf.explode()

            # Plot the geometry
            gdf.plot(ax=ax, 
                     facecolor=gdf['color'], 
                     edgecolor='none', 
                     alpha=alfa_clust,
                     zorder=1)
            gdf.plot(ax=ax, 
                     facecolor='none', 
                     edgecolor=gdf['color'], 
                     alpha=1, 
                     linewidth=2, 
                     zorder=1)
            if plot_labels:
                # Iterate over each polygon in the GeoDataFrame
                for idx, row in gdf.iterrows():
                    
                    if len(clusters_highlight)!=0:
                        if row["cluster"] not in clusters_highlight:
                            continue
                    
                    if auto_adjust_text:
                    
                        polygon = row['geometry']
                        label = row['label']
                        
                        # calculate the minimum bounding box of the polygon
                        mbb = Polygon(polygon.exterior).minimum_rotated_rectangle

                        # Get the minimum rotated rectangle of the bounding box
                        rotated_rect = mbb.minimum_rotated_rectangle

                        # Get the angle of the major axis of the minimum rotated rectangle
                        angle = np.rad2deg(np.arctan2(rotated_rect.bounds[3]-rotated_rect.bounds[1],
                                                      rotated_rect.bounds[2]-rotated_rect.bounds[0]))

                        x, y = mbb.representative_point().coords[0]
                        ax.plot(*mbb.exterior.xy, color="red")
                        mbb_coords = mbb.exterior.coords
                        
                        # calculate the aspect ratio of the MBB
                        mbb_width = Point(mbb_coords[0]).distance(Point(mbb_coords[1]))
                        mbb_height = Point(mbb_coords[1]).distance(Point(mbb_coords[2]))

                        if mbb_width<mbb_height:
                            save_wid = mbb_width
                            mbb_width = mbb_height
                            mbb_height = save_wid
                        
                        plt.ioff()
                        # calculate the aspect ratio of the label text
                        label_width, label_height = ax.text(0, 0, label, ha='left', va='bottom', fontsize=clusterout_maxtext_size).get_window_extent().size
                        plt.ion()

                        # scale the label text to fit inside the MBB
                        if label_width>mbb_width:
                            scale_factor = mbb_width/label_width
                        else:
                            scale_factor = 1

                        fontsize = clusterout_maxtext_size * scale_factor*10
                        
                        
                        ax.text(x, 
                                y, 
                                label, 
                                ha='center', 
                                va='center', 
                                fontsize=fontsize, 
                                rotation=angle, 
                                color='white', 
                                weight='bold')
                    else:
                        # Get the centroid of the polygon
                        centroid = row.geometry.centroid
                        
                        # Get the label
                        label = row['label']
                        
                        # Create a text object with the label
                        ax.text(x=centroid.x+0.05, 
                                y=centroid.y+0.05, 
                                s=label, 
                                ha='center', 
                                va='center', 
                                color='black', 
                                alpha=0.7, 
                                weight='bold', 
                                fontsize=clusterout_maxtext_size)
                        
                        ax.text(x=centroid.x, 
                                y=centroid.y, 
                                s=label, 
                                ha='center', 
                                va='center', 
                                color="white", 
                                weight='bold', 
                                fontsize=clusterout_maxtext_size)
                        


        # Plotting Parameters
        ax.set_xlim(-0.6-0.5, 2*self.mapsize[0]-0.5+0.6)
        ax.set_ylim(-0.5660254-0.81, 2*self.mapsize[1]*0.8660254-2*0.560254+0.75)
        ax.set_axis_off()
        ax.invert_yaxis()

        plt.title(f"Clustering Matrix - {clusters.max()} clusters",
                  horizontalalignment='center',  
                  verticalalignment='top', 
                  size=title_size, 
                  pad=title_pad)
        
        if max_clust <=10:
            ax2 = f.add_subplot(gs[20:80, 90:])
        elif max_clust<=20:
            ax2 = f.add_subplot(gs[20:80, 80:])
        elif max_clust<=30:
            ax2 = f.add_subplot(gs[20:80, 70:])
        else:
            ax2 = f.add_subplot(gs[20:80, 60:])
        
        
        ax2.invert_yaxis()
        ax2.set_aspect('equal')

        n_cols = int(np.ceil(max_clust/10))
        n_rows = int(np.ceil(max_clust / n_cols))

        hex_height = 0.096
        pad = (0.1-hex_height)
        radius =  hex_height/2
        total_height = hex_height * n_rows + n_rows * pad
        shift = hex_height
        y_start = ((1 - total_height) / 2)+shift/2
        x_start = pad+shift/2
        text_pad = hex_height*3

        condition = max_clust-1 if watermark_neurons else max_clust

        for i, (xfac, yfac) in enumerate(np.ndindex((n_cols, n_rows))):
            if i+1 <= condition:
                cluster = i+1 if len(clusters_highlight) == 0 else clusters_highlight[i]
                x_center = x_start+(xfac)*shift+xfac*pad+xfac*text_pad
                y_center = y_start+(yfac)*shift+yfac*pad

                if len(clusters_highlight) == 0:
                    color = cmap(norm(cluster))
                else:
                    color = "gray" if cluster not in clusters_highlight else cmap(norm(cluster))

                hex_points = RegularPolygon((x_center, y_center), 
                                            numVertices=6, 
                                            radius=radius,
                                            facecolor=color, 
                                            edgecolor=None, 
                                            alpha=alfa_clust)
                ax2.add_patch(hex_points)

                hex_points = RegularPolygon((x_center, y_center), 
                                            numVertices=6, 
                                            radius=radius-radius*0.05,
                                            facecolor=None,
                                            fill=False, 
                                            edgecolor=color,
                                            linewidth=2)
                ax2.add_patch(hex_points)
                if len(custom_labels)>0:
                    cluster_name = custom_labels[cluster-1]
                else:
                    cluster_name = f"Cluster #{cluster}"
                ax2.annotate(cluster_name,
                            xy=(x_center+radius+0.01, y_center),
                            xytext=(0, 0),
                            textcoords="offset points",
                            color='black',
                            weight='bold',
                            fontsize=legend_text_size,
                            ha='left',
                            va='center')
            else:
                if watermark_neurons:
                    x_center = x_start+(xfac)*shift+xfac*pad+xfac*text_pad
                    y_center = y_start+(yfac)*shift+yfac*pad

                    hex_points = RegularPolygon((x_center, y_center), 
                                            numVertices=6, 
                                            radius=radius-radius*0.05,
                                            facecolor="White",
                                            fill = True, 
                                            edgecolor="Black",
                                            linewidth=2)
                    
                    ax2.add_patch(hex_points)

                    ax2.annotate(f"#",
                            xy=(x_center, y_center),
                            xytext=(0, 0),
                            textcoords="offset points",
                            color='black',
                            weight='bold',
                            fontsize=legend_text_size,
                            ha='center',
                            va='center')

                    ax2.annotate(f"Neuron Number",
                            xy=(x_center+radius+0.01, y_center),
                            xytext=(0, 0),
                            textcoords="offset points",
                            color='black',
                            weight='bold',
                            fontsize=legend_text_size,
                            ha='left',
                            va='center')
                    break

        
        ax2.set_title(legend_title if legend_title!=False else "Legend", 
                      fontdict={"fontsize": legend_title_size},
                      loc="center", 
                      pad=5,
                      fontweight='bold',
                      y=1-y_start+0.03)
        
        ax2.set_xlim(0, n_cols*hex_height+n_cols*pad+n_cols*text_pad)
        ax2.set_ylim(1, -0.01)
        
        ax2.set_axis_off()

        #ADD WATERMARK
        # Add white space subplot below the plot
        ax3 = f.add_subplot(gs[95:100, 0:20], zorder=-1)

        # Add the watermark image to the white space subplot
        ax3.imshow(self.foot, aspect='equal', alpha=1)
        ax3.axis('off')

        f.subplots_adjust(wspace=0.1)

        if save:
            if file_path:
                f.savefig(f"{file_path}/{file_name}.jpg",dpi=300, bbox_inches = "tight")
            else:
                path = 'Plots/Clusters'
                os.makedirs(path, exist_ok=True)

                f.savefig(f"Plots/Clusters/{file_name}.jpg",dpi=300, bbox_inches = "tight")
        
        if return_geodataframe:
            return gdf
                

    def generate_hex_lattice(self, n_columns, n_rows):
        """
        Generates the xy coordinates of the BMUs for an odd-r hexagonal grid.
        Args:
            n_rows: Number of lines in the Kohonen map.
            n_columns: Number of columns in the Kohonen map.

        Returns:
            Coordinates of the [x,y] format for the BMUs in a hexagonal grid.

        """
        ratio = np.sqrt(3) / 2

        coord_x, coord_y = np.meshgrid(np.arange(n_columns),
                                       np.arange(n_rows), 
                                       sparse=False, 
                                       indexing='xy')
        coord_y = coord_y * ratio
        coord_x = coord_x.astype(float)
        coord_x[1::2, :] += 0.5
        coord_x = coord_x.ravel()
        coord_y = coord_y.ravel()

        coordinates = np.column_stack([coord_x, coord_y])

        return coordinates
    

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
