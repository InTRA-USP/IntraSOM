import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.patches import RegularPolygon
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import geopandas as gpd
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
import pandas as pd
import importlib.resources as resources
from PIL import Image
from .object_functions import NormalizerFactory

class ClusterRepresentativity(object):

    """
    This module helps in the assessment of the representativity of the SOM and K-Means clustering.

    It focuses on two main representativity notions:
    - Fair clusters: following the principle of individual fairness (unsupervised fairness), where no group is significantly under-represented, we calculate coverage fairness and representativity fairness given the samples, samples's cluster labels and clusters's representatives.
    - Typical samples: select typical samples (individual objects) that best characterize each cluster or population.

    The module works after running the "kmeans" function from the clustering module. All analyses are based on the normalized values of data, SOM neurons, and K-Means centroids.

    Methods:
        calculate_neurons_cmpf: calculates the values of the Cluster Membership Probability Function (CMPF) for all neurons. The CMPF is:
            - a measure of the representativity of each neuron (i) to each cluster (k)
            - uses the similarity between SOM neurons and K-means centroids weighted by the density of samples in a given neuron
            - if similarity is a metric that varies from 0 to 1, CMPF will vary from 0 to n(i)/S, with n(i) being the maximum number of hits of a neuron i in a cluster
            - used to analyze clusters representatives, which are used for measuring fairness, ordering clusters and selecting typical samples

        cmpf_jitter_plot: plot the distribution of CMPF values for each cluster in a jitterplot

        cmpf_som_plot: plot the distribution of maximum CMPF values of each neuron in the SOM latent space, highlighting the clusters representatives.

        symmetry_index: this is a measure of coverage fairness that quantifies the spatial coverage of clusters representatives. This measures ranges from 0 to 1, with higher values indicating a more regular (fair) distribution of representatives in the data space.

        jain_index: this is a measure of representativity fairness that, according to Padmanabhan & Abraham (2020), quantifies the degree of egalitarianism of the formed clusters. This measure ranges from 0 to 1, where higher values indicate more uniform (fair) clusters.

        typical_sample_analysis: gets samples scores used to select typical samples of each cluster based on the CMPF values, the Silhouette Coefficient (SC) and the samples-BMU distances.
            - The SC is a measure of how compact each sample is in respect to its own cluster and how well separated it is in respect to its nearest cluster. SC values range from -1 to +1, where negative values indicate overlapping clusters and positive values indicate that data points are much closer to their own cluster than other clusters. It can be used solely to select typical samples of clusters or associated with the CMPF criterion.
    """

    def __init__(self, cluster_object):

        self.som_object = cluster_object.som_object

        # The loaded parameters (samples, neurons and centroids) are normalized
        self._data = cluster_object.som_object._data
        self._neurons = cluster_object.som_object.codebook.matrix
        self._clusters_centroids = cluster_object._clusters_centroids

        # Assignments and number of hits
        self.samples_bmu = cluster_object.som_object.results_dataframe['Neuron'] # pandas series, the index is added by 1
        self._neurons_labels = cluster_object._neurons_labels # neurons clusters labels (ranging from 1 to n_clusters)
        self._samples_labels = cluster_object._samples_labels # samples clusters labels (ranging from 0 to n_clusters-1)
        self.bmu_hits = cluster_object.som_object.results_dataframe['Neuron'].value_counts() # pandas series, the index is added by 1

        # For plotting
        self.mapsize = cluster_object.mapsize # (n_columns, n_rows)
        image_file = resources.files('intrasom').joinpath('images/foot.jpg')
        self.foot = Image.open(image_file)

    def calculate_neurons_cmpf(self,
                               similarity: str = 'min-max',
                               epsilon: float = 1e-10,
                               rounded: bool = False,
                               decimals: int = 3):
        """
        This function returns a dataframe of shape "number of neurons x number of clusters" in which each cell contains the Cluster Membership Probability Function (CMPF) values.

        The CMPF is a function that uses the parameters of the SOM and K-Means clustering framework to calculate the degree to which each neuron, more specifically which BMU, is associated to a given cluster.

        To calculate the CMPF, a similarity metric between neurons and centroids (sim(i,k)) is weighted by the number of hits of each neuron (n(i)) divided by the total number of samples in the dataset (S): sim(i,k) * (n(i)/S).

        Then, CMPF values are normalized across clusters for each neuron, such that for each BMU, the probabilities sum up to 1.0. In this way, neurons with no hits are assigned a CMPF value of 0.0.

        Args:
            similarity (str): similarity metric used to transform the distances between neurons and centroids into similarities.
            epsilon (float): when calculating the Min-Max similarities, sum this factor to avoid division by 0.
            rounded (bool): returns the rounded CMPF values.
            decimals (int): decimals used for rounding the CMPF values.

        Sets:
            neurons_similarities (np.ndarray): the similarity of each neuron in respect to each cluster (n_neurons, n_clusters).
            cluster_representatives_indices (np.ndarray): the indices of the neurons with maximum CMPF value of each cluster (n_clusters).
            clusters_weights (np.ndarray): the weights of each cluster calculated by normalizing the means of the neurons CMPF values. These weights are probabilities that represent the ordering of representativity of each cluster and their sum for all clusters is equal to 1.0.

        Returns:
            np.ndarray: CMPF array of shape (n_neurons, n_clusters), with cluster membership probabilities for each neuron.
        """

        if similarity == 'min-max':
            # Calculate the neurons similarities to the centroids by normalizing across the rows
            # neurons_similarities is of shape (n_neurons, n_clusters)
            neurons_dists = np.linalg.norm(self._neurons[:, np.newaxis, :] - self._clusters_centroids, axis=2)
            nd_min = np.min(neurons_dists, axis=1, keepdims=True)
            nd_max = np.max(neurons_dists, axis=1, keepdims=True)
            neurons_similarities = 1 - (neurons_dists - nd_min) / (nd_max - nd_min + epsilon)
            self.neurons_similarities = neurons_similarities
        else:
            print('Similarity metric not implemented.')

        # Weight the similarities by the (number of hits)/(number of samples)
        # bmu_hits is of shape (n_bmu)
        # neurons_hits and hits_weighting_factor are of shape (n_neurons)
        # unnorm_cmpf is of shape (n_neurons, n_clusters)
        n_samples = self._data.shape[0]
        n_neurons = self._neurons.shape[0]
        bmu_hits = self.bmu_hits
        neurons_hits = np.zeros(n_neurons, dtype=int)
        neurons_hits[bmu_hits.index - 1] = bmu_hits.values
        hits_weighting_factor = neurons_hits/n_samples
        unnorm_cmpf = neurons_similarities * hits_weighting_factor[:, np.newaxis] # (n_neurons) > (n_neurons,)

        # Normalize the CMPF values to get the responsibilities for each neuron (ranging from 0.0 to 1.0)
        # row_sums is of shape (n_neurons)
        # cmpf is of shape (n_neurons, n_clusters)
        row_sums = unnorm_cmpf.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cmpf = unnorm_cmpf / row_sums
        self.cmpf = cmpf

        # For instance, it is useful to get:
        # 1. The weight of each cluster considering the mean of the CMPF values
        # 2. The cluster representative or "typical neuron" with the maximum CMPF value to the cluster;
        # cluster_representatives_indices and clusters_weights are of shape (n_clusters)
        cluster_representatives_indices = np.argmax(cmpf, axis=0)
        clusters_weights = np.mean(cmpf, axis=0)
        clusters_weights /= clusters_weights.sum()

        # Get the clusters ordering through the mean CMPF values (clusters_weights)
        cluster_weights_sorted_indices = np.argsort(-clusters_weights) # the order of indices of the cluster_weights from largest to smallest
        clusters_order = np.zeros_like(clusters_weights, dtype=int)
        for rank_num, original_idx in enumerate(cluster_weights_sorted_indices):
            clusters_order[original_idx] = rank_num + 1
        self.clusters_order = clusters_order

        # Representatives indices starting at 1
        self.cluster_representatives_indices = cluster_representatives_indices+1
        self.clusters_weights = clusters_weights

        return np.round(cmpf, decimals=decimals) if rounded else cmpf

    def cmpf_jitter_plot(self,
                        plot_height: int = 6,
                        plot_width: float = 12,
                        colormap: str = "gist_rainbow",
                        neurons_max_size: float = 120,
                        alpha_neuron_legend: float = 0.7,
                        watermark_neurons: bool = True,
                        y_watermark_neurons_text: float = 0,
                        watermark_neurons_fontsize: float = 5,
                        title: str = "Jittering strip plot with CMPF values per cluster and means",
                        title_size: float = 12,
                        xlabel_size: float = 10,
                        ylabel_size: float = 10,
                        xlabel: str ="Cluster",
                        ylabel: str ="CMPF values",
                        clusters_highlight: list = [],
                        yrange: list = [0.0,1.0],
                        legend_title: str = "Clusters weights\nranking (mean CMPF)",
                        custom_labels: list = [],
                        legend_loc: str = "center",
                        legend_title_size: float = 8,
                        legend_text_size: float = 6,
                        num_hexa_dist_factor: float = 0.015,
                        hexa_label_dist_factor: float = 0.025):

        """
        For visualization of the distributions of CMPF values for each neuron. For visualizing the distributions of the maximum CMPF values for each neuron in the SOM latent space, use "cmpf_som_plot".
        """

        try:
            cmpf = self.cmpf
        except:
            self.calculate_neurons_cmpf() # calculate the CMPF
            cmpf = self.cmpf

        n_clusters = cmpf.shape[1]

        # Plot CMPF values to check ties on each cluster
        f = plt.figure(figsize=(plot_width, plot_height), dpi=300)
        gs_height = plot_height*10
        gs_width = plot_width*10
        gs = gridspec.GridSpec(gs_height, gs_width)
        if n_clusters <=10:
            ax1 = f.add_subplot(gs[:int(0.90*gs_height), 0:int(0.89*gs_width)])
        elif n_clusters <=20:
            ax1 = f.add_subplot(gs[:int(0.90*gs_height), 0:int(0.79*gs_width)])
        elif n_clusters <=30:
            ax1 = f.add_subplot(gs[:int(0.90*gs_height), 0:int(0.69*gs_width)])
        else:
            ax1 = f.add_subplot(gs[:int(0.90*gs_height), 0:int(0.59*gs_width)])

        # Jittering strip plot
        if clusters_highlight == []:
            clusters_indices = np.arange(n_clusters)
        else:
            clusters_indices = np.array(clusters_highlight) - 1
        cmap = plt.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, cmpf.shape[1]))
        for i, cluster_label in enumerate(clusters_indices):
            # Plot the neurons
            y = cmpf[:, cluster_label]
            x = np.random.normal(i + 1, 0.08, size=len(y))
            # Plot the mean line of each cluster
            mean_y = np.mean(y[y > 0])  # ignore neurons with CMPF = 0
            ax1.hlines(mean_y, i + 0.7, i + 1.3, colors=colors[cluster_label], linestyles='dashed', linewidth=2, zorder=2)
            # Plot the neurons
            ax1.scatter(x, y, color=colors[cluster_label], marker='h', alpha=y, edgecolor=colors[cluster_label], s=neurons_max_size*y, zorder=3)
            if watermark_neurons:
                for xi, yi, neuron_idx in zip(x, y, range(1, len(y)+1)):
                    y_text = yi + y_watermark_neurons_text
                    if yi > mean_y and (yrange[0] <= y_text <= yrange[1]):  # only show for neurons above the mean line and inside the y range
                        ax1.text(xi, yi + y_watermark_neurons_text, str(neuron_idx),
                                fontsize=watermark_neurons_fontsize,
                                ha='center', va='center', color='black')

        ax1.set_ylim(yrange)
        ax1.set_xticks(np.arange(1, len(clusters_indices) + 1))
        ax1.set_xticklabels([f'#{i+1}' for i in clusters_indices])
        ax1.set_xlabel(xlabel, fontsize=xlabel_size)
        ax1.set_ylabel(ylabel, fontsize=ylabel_size)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title(title, fontsize=title_size)

        # Plot legend
        legend_top = int(0.3*gs_height)
        legend_bottom = int(0.7*gs_height)
        if n_clusters <=10:
            ax2 = f.add_subplot(gs[legend_top:legend_bottom, int(0.90*gs_width):])
        elif n_clusters<=20:
            ax2 = f.add_subplot(gs[legend_top:legend_bottom, int(0.80*gs_width):])
        elif n_clusters<=30:
            ax2 = f.add_subplot(gs[legend_top:legend_bottom, int(0.70*gs_width):])
        else:
            ax2 = f.add_subplot(gs[legend_top:legend_bottom, int(0.60*gs_width):])
        ax2.invert_yaxis()
        ax2.set_aspect('equal')

        # Legend layout parameters
        n_cols = int(np.ceil(n_clusters/10))
        n_rows = int(np.ceil(n_clusters / n_cols))
        hex_height = 0.096
        pad = (0.1-hex_height)
        radius =  hex_height/2
        total_height = hex_height * n_rows + n_rows * pad
        shift = hex_height
        y_start = ((1 - total_height) / 2)+shift/2
        x_start = pad+shift/2
        text_pad = hex_height*3

        # Fill legend layout
        clusters_order = self.clusters_order
        for i, (xfac, yfac) in enumerate(np.ndindex((n_cols, n_rows))):
            x_center = x_start+(xfac)*shift+xfac*pad+xfac*text_pad
            y_center = y_start+(yfac)*shift+yfac*pad
            ax2.annotate(f'{i+1}.',
                xy=(x_center, y_center),
                xytext=(0, 0),
                textcoords="offset points",
                color='black',
                weight='bold',
                fontsize=legend_text_size,
                ha='left',
                va='center')

            clust_idx = np.where(clusters_order == i+1)[0][0]
            hex_points = RegularPolygon((x_center+radius+num_hexa_dist_factor*legend_text_size, y_center), 
                numVertices=6, 
                radius=radius-radius*0.05,
                facecolor=colors[clust_idx],
                fill = True,
                alpha=alpha_neuron_legend,
                edgecolor=colors[clust_idx],
                linewidth=2)
            ax2.add_patch(hex_points)

            if custom_labels != []:
                ax2.annotate(custom_labels[clust_idx],
                    xy=(x_center+1.2*radius+hexa_label_dist_factor*legend_text_size, y_center),
                    xytext=(0, 0),
                    textcoords="offset points",
                    color='black',
                    weight='bold',
                    fontsize=legend_text_size,
                    ha='left',
                    va='center')
            else:
                ax2.annotate(f'Cluster #{clust_idx+1}',
                    xy=(x_center+1.2*radius+hexa_label_dist_factor*legend_text_size, y_center),
                    xytext=(0, 0),
                    textcoords="offset points",
                    color='black',
                    weight='bold',
                    fontsize=legend_text_size,
                    ha='left',
                    va='center')

        ax2.set_title(legend_title, 
                      fontdict={"fontsize": legend_title_size},
                      loc=legend_loc, 
                      pad=5,
                      fontweight='bold',
                      y=1-y_start+0.03)
        
        ax2.set_xlim(0, n_cols*hex_height+n_cols*pad+n_cols*text_pad)
        ax2.set_ylim(1, -0.01)
        
        ax2.set_axis_off()

        # Add watermark
        # Add white space subplot below the plot
        image_width = 4*(gs_height-int(0.95*gs_height))
        ax3 = f.add_subplot(gs[int(0.95*gs_height):gs_height, 0:image_width], zorder=-1)

        # Add the watermark image to the white space subplot
        ax3.imshow(self.foot, aspect='equal', alpha=1)
        ax3.axis('off')

        plt.show()

    def cmpf_som_plot(self,
                      figsize: tuple = (16,14),
                      watermark_neurons: bool = False,
                      watermark_typical_neurons: bool = True,
                      neurons_fontsize: float = 12,
                      colormap: str = "gist_rainbow",
                      clusters_highlight: list = [],
                      legend_title: bool = False,
                      legend_title_size: float = 12,
                      legend_text_size: float = 10,
                      title: str = "Maximum CMPF value of each SOM neuron",
                      plot_labels: bool = True,
                      auto_adjust_text: bool = True,
                      clusterout_maxtext_size: float = 12,
                      alfa_clust_legend: float = 0.5,
                      title_size: float = 25,
                      title_pad: float = 40,
                      custom_labels: list = [],
                      cluster_outline: bool = True,
                      neuron_number_label: str = "Neuron Number",
                      typical_neuron_label: str = "Typical Neuron"):
        """
        Plots a SOM graph showing the maximum CMPF value of each neuron for its assigned cluster (equivalent to the neuron responsibility, the higher the CMPF value, more typical). Also, highlights the typical neuron of each cluster.
        """
        
        neurons_labels = self._neurons_labels
        clusters = neurons_labels.reshape(self.mapsize[1], self.mapsize[0])

        try:
            cmpf = self.cmpf
        except:
            self.calculate_neurons_cmpf() # calculate the CMPF
            cmpf = self.cmpf
        max_cmpf = np.max(cmpf, axis=1).reshape(self.mapsize[1], self.mapsize[0])

        f = plt.figure(figsize=figsize, dpi=300)
        plot_height = int(90*(self.mapsize[1]/self.mapsize[0]))
        gs_height = plot_height + 5
        gs = gridspec.GridSpec(gs_height, 100)

        max_clust = neurons_labels.max() + 1 # including the typical neuron
        max_clust = max_clust+1 if watermark_neurons else max_clust

        pad_subplots = 3
        if max_clust <=10:
            ax = f.add_subplot(gs[:plot_height, :90-pad_subplots])
        elif max_clust<=20:
            ax = f.add_subplot(gs[:plot_height, :80-pad_subplots])
        elif max_clust<=30:
            ax = f.add_subplot(gs[:plot_height, :70-pad_subplots])
        else:
            ax = f.add_subplot(gs[:plot_height, :60-pad_subplots])

        ax.set_aspect('equal')

        xx = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,0], (self.mapsize[1], self.mapsize[0]))
        yy = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,1], (self.mapsize[1], self.mapsize[0]))

        norm = mpl.colors.Normalize(vmin=np.nanmin(neurons_labels), vmax=np.nanmax(neurons_labels))

        cmap = cm.get_cmap(colormap)

        for j in range(clusters.shape[0]):
            for i in range(clusters.shape[1]):
                nnodes = self.mapsize[0] * self.mapsize[1]
                grid_neurons = np.linspace(1,nnodes, nnodes).reshape(self.mapsize[1], self.mapsize[0])
                color = cmap(norm(clusters[j][i]))
                hexagon = RegularPolygon(
                    (xx[(j, i)]*2, yy[(j,i)]*2), 
                    numVertices=6, 
                    radius=(2/np.sqrt(3)-0.04*(2/np.sqrt(3)))*max_cmpf[j][i],
                    facecolor = color,
                    edgecolor="black" if grid_neurons[j,i] in self.cluster_representatives_indices else None,
                    linewidth=2 if grid_neurons[j,i] in self.cluster_representatives_indices else 0, 
                    alpha = 1 if grid_neurons[j,i] in self.cluster_representatives_indices else max_cmpf[j][i], 
                    zorder=1)
                ax.add_patch(hexagon)

                if watermark_neurons:
                    hexagon = RegularPolygon(
                        (xx[(j, i)]*2, yy[(j,i)]*2),
                        numVertices=6,
                        radius=2/np.sqrt(3),
                        facecolor= "white",
                        edgecolor='black',
                        alpha=0.1, 
                        zorder=2)
                    ax.add_patch(hexagon)
                    ax.text(xx[(j,i)]*2, yy[(j,i)]*2, 
                        s=f"{int(grid_neurons[j,i])}", 
                        size = neurons_fontsize,
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        color='black', 
                        zorder=2)
                else:
                    if watermark_typical_neurons:
                        ax.text(xx[(j,i)]*2, yy[(j,i)]*2, 
                            s=f"{int(grid_neurons[j,i]) if grid_neurons[j,i] in self.cluster_representatives_indices else ""}", 
                            size = neurons_fontsize,
                            horizontalalignment='center', 
                            verticalalignment='center', 
                            color='black', 
                            zorder=2)

        if cluster_outline:
            cluster_vertices_dict = {}

            for j in range(clusters.shape[0]):
                for i in range(clusters.shape[1]):

                    label = clusters[j][i]

                    if len(clusters_highlight) == 0:
                        color = cmap(norm(label))
                    else:
                        color = None if clusters[j][i] not in clusters_highlight else cmap(norm(label))
                    
                    if label not in cluster_vertices_dict:
                        cluster_vertices_dict[label] = []

                    # Get the vertices of the hexagon
                    hexagon = RegularPolygon((xx[(j, i)]*2, yy[(j,i)]*2), 
                                        numVertices=6, 
                                        radius=2/np.sqrt(3)+0.04,
                                        facecolor = color, 
                                        alpha = alfa_clust_legend)
                    vertices = hexagon.get_verts()
                    polygon = Polygon(vertices)
                    cluster_vertices_dict[label].append(polygon)

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
                    color = None if label not in clusters_highlight else cmap(norm(label))
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
                     facecolor='none', 
                     edgecolor=gdf['color'], 
                     alpha=1, 
                     linewidth=2, 
                     zorder=3)
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
                                color='black', 
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
                                color='gray', 
                                alpha=0.7, 
                                weight='bold', 
                                fontsize=clusterout_maxtext_size)
                        
                        ax.text(x=centroid.x, 
                                y=centroid.y, 
                                s=label, 
                                ha='center', 
                                va='center', 
                                color="black", 
                                weight='bold', 
                                fontsize=clusterout_maxtext_size)

        # Plotting Parameters
        ax.set_xlim(-0.6-0.5, 2*self.mapsize[0]-0.5+0.6+0.5)
        ax.set_ylim(-0.5660254-0.81, 2*self.mapsize[1]*0.8660254-2*0.560254+0.75+0.2886751)
        ax.set_axis_off()
        ax.invert_yaxis()

        plt.title(title,
                  horizontalalignment='center',  
                  verticalalignment='top', 
                  size=title_size, 
                  pad=title_pad)

        # Plot legend
        legend_top = int(0.2*gs_height)
        legend_bottom = int(0.8*gs_height)
        if max_clust <=10:
            ax2 = f.add_subplot(gs[legend_top:legend_bottom, 90:])
        elif max_clust<=20:
            ax2 = f.add_subplot(gs[legend_top:legend_bottom, 80:])
        elif max_clust<=30:
            ax2 = f.add_subplot(gs[legend_top:legend_bottom, 70:])
        else:
            ax2 = f.add_subplot(gs[legend_top:legend_bottom, 60:])

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
            if i+1 <= condition-1:
                cluster = i+1
                x_center = x_start+(xfac)*shift+xfac*pad+xfac*text_pad
                y_center = y_start+(yfac)*shift+yfac*pad

                color = cmap(norm(cluster))

                hex_points = RegularPolygon((x_center, y_center), 
                                            numVertices=6, 
                                            radius=radius,
                                            facecolor=color, 
                                            edgecolor=None, 
                                            alpha=alfa_clust_legend)
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

                ax2.annotate(typical_neuron_label,
                        xy=(x_center+radius+0.01, y_center),
                        xytext=(0, 0),
                        textcoords="offset points",
                        color='black',
                        weight='bold',
                        fontsize=legend_text_size,
                        ha='left',
                        va='center')

                if watermark_neurons or watermark_typical_neurons:
                    hex_points = RegularPolygon((x_center, y_start+(yfac+1)*shift+yfac*pad), 
                                            numVertices=6, 
                                            radius=radius-radius*0.05,
                                            facecolor="White",
                                            fill = True, 
                                            edgecolor="Gray",
                                            linewidth=1)
                    
                    ax2.add_patch(hex_points)

                    ax2.annotate(f"#",
                            xy=(x_center, y_start+(yfac+1)*shift+yfac*pad),
                            xytext=(0, 0),
                            textcoords="offset points",
                            color='black',
                            weight='bold',
                            fontsize=legend_text_size,
                            ha='center',
                            va='center')

                    ax2.annotate(neuron_number_label,
                            xy=(x_center+radius+0.01, y_start+(yfac+1)*shift+yfac*pad),
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

        # Add watermark
        # Add white space subplot below the plot
        image_width = 4*(gs_height-plot_height)
        ax3 = f.add_subplot(gs[plot_height:gs_height, 0:image_width], zorder=-1)

        # Add the watermark image to the white space subplot
        ax3.imshow(self.foot, aspect='equal', alpha=1)
        ax3.axis('off')

    @staticmethod
    def symmetry_index(representatives: np.array):
        """
        Compute the Symmetry index for the clustering result. The Symmetry index is a measure of coverage fairness and is calculated considering the set of nearest-neighbor distances of representatives. Then Symmetry is obtained in terms of the mean and standard deviation of this set of nearest-neighbor distances. Its values ranges from 0 to 1, with higher values indicating greater spatial coverage of representatives.

        Args:
            representatives (np.array): representatives (n_clusters, n_dims)

        Returns:
            symmetry (float): Jain index
        """

        # Calculate the degree to which representatives follow a homogeneous distribution/uniform coverage
        representatives_pairwise_dists = np.linalg.norm(representatives[:, np.newaxis, :] - representatives, axis=2)
        nearest_neighbor_distances = []

        # Find the nearest representative distance
        for i in range(representatives.shape[0]):
            distances_from_i = representatives_pairwise_dists[i, :]
            min_distance = np.min(distances_from_i[distances_from_i > 0]) # calculate distance to the nearest neighbor
            nearest_neighbor_distances.append(min_distance)

        # Calculate mean and standard deviation of the nearest neighbor distances
        nearest_neighbor_distances = np.array(nearest_neighbor_distances)
        mean_nn_dist = np.mean(nearest_neighbor_distances)
        std_nn_dist = np.std(nearest_neighbor_distances)

        # Calculate Symmetry index
        if mean_nn_dist == 0: # avoid division by 0 (shouldn't happen)
            symmetry = np.nan
        else:
            symmetry = mean_nn_dist / (mean_nn_dist + std_nn_dist) # values closer to 1, mean a fair coverage

        return symmetry

    @staticmethod
    def jain_index(X: np.array, labels: np.array, centroids: np.array):
        """
        Compute the Jain index for the clustering result. The Jain index is calculated using the distances between samples and centroids (intra-cluster distances) and evaluates the degree of egalitarianism, meaning it favors clusterings where intra-cluster distances are distributed homogeneously across clusters. It ranges from 0 to 1, where values closer to 1 indicate a higher degree of egalitarianism.

        Args:
            X (np.array): samples (n_samples, n_dims)
            labels (np.array): clusters labels of the samples (n_samples)
            centroids (np.array): centroids (n_clusters, n_dims)

        Returns:
            jain (float): Jain index
        """

        dists = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        dists_to_representatives = dists[np.arange(dists.shape[0]), labels]
        jain = np.square(np.sum(dists_to_representatives)) / (dists_to_representatives.size * np.sum(np.square(dists_to_representatives)))

        return jain

    def typical_sample_analysis(self):
        """
        This function performs a typical sample analysis based on the selected neurons as cluster representatives. Three main criterions are followed: the CMPF for selecting representatives and the Silhouette Coefficient (SC) and samples-BMU distances to select typical samples. Given that, this method implement three typical samples selection criteria:
            1. Samples with highest SC of each cluster (sc_typical_samples_indices)
            2. Samples within the representatives proto-cluster closest to the clusters representatives (cmpf_dists_typical_samples_indices)
            3. Samples within the representatives proto-cluster with highest SC (cmpf_sc_typical_samples_indices)
        Where all indices of selected samples starts at 0.
        
        Sets:
            sc_typical_samples_indices (np.array): the indices of the typical samples selected only based on the SC of the samples considering its clusters assignments.
            cmpf_dists_typical_samples_indices (np.array): the indices of the typical samples selected based on the representatives and their distances to the samples in the dataset
            cmpf_sc_typical_samples_indices (np.array): the indices of the typical samples selected based on the representatives and the SC of the samples associated to the representatives.
        """

        samples = self._data
        neurons = self._neurons
        n_samples = len(samples)
        labels_neurons = self.samples_bmu.values-1 # indices of samples starting at 0
        labels_clusters = self._samples_labels # indices of samples starting at 0
        cmpf = self.cmpf

        # cmpf_values is of shape (n_neurons) and silhouette_values and samples_bmu_dists are of shape (n_samples)
        cmpf_values = np.array([cmpf[labels_neurons[i], labels_clusters[i]] for i in range(n_samples)])
        silhouette_values = silhouette_samples(samples, labels_clusters)
        samples_bmu_dists = np.min(np.linalg.norm(samples[:, np.newaxis, :] - neurons, axis=2), axis=1)

        # Create the pandas dataframe (n_samples, 6)
        samples_scores = pd.DataFrame({
            'Sample_index': np.arange(n_samples),
            'BMU': labels_neurons+1,
            'Cluster': labels_clusters+1,
            'CMPF_value': cmpf_values,
            'Silhouette': silhouette_values,
            'Samples_BMU_dists': samples_bmu_dists
        })

        # Get the clusters ordering through the mean CMPF values (clusters_weights)
        clusters_order = self.clusters_order

        # Get the typical samples based solely on the SC
        sc_typical_samples_indices = samples_scores.groupby('Cluster')['Silhouette'].idxmax()
        # Get the typical samples based on CMPF plus a second criterion
        samples_representatives_scores = samples_scores[samples_scores['BMU'].isin(self.cluster_representatives_indices)]
        cmpf_dists_typical_samples_indices = samples_representatives_scores.groupby('BMU')['Samples_BMU_dists'].idxmin()
        cmpf_sc_typical_samples_indices = samples_representatives_scores.groupby('BMU')['Silhouette'].idxmax()

        self.sc_typical_samples_indices = sc_typical_samples_indices
        self.cmpf_dists_typical_samples_indices = cmpf_dists_typical_samples_indices
        self.cmpf_sc_typical_samples_indices = cmpf_sc_typical_samples_indices

        # Order the samples on the pandas dataframe based on clusters weights obtained by the mean values of CMPF, then on CMPF values inside each cluster
        samples_scores['Cluster_order'] = samples_scores['Cluster'].apply(lambda c: clusters_order[c - 1])
        # samples_scores = samples_scores.sort_values(
        #     by=['Cluster_order', 'CMPF_value', 'Silhouette'],
        #     ascending=[True, False, False]
        # ).reset_index(drop=True)
        # samples_scores.drop(['Cluster_order'], axis=1, inplace=True)
        self.samples_scores = samples_scores

        return samples_scores

    def silhouette_plot(
            self,
            figsize: tuple = (6,5),
            title: str = "Silhouette Plot (ordered by cluster importance)",
            xlabel: str = "Silhouette values",
            ylabel: str = "Samples (by cluster order)",
            colormap: str = "gist_rainbow",
            custom_labels: list = [],
            legend_upper_left_bbox_anchor: tuple = (1.02,0.7),
            text_fontsize: float = 14,
            labels_fontsize: float = 12,
            xy_labels_fontsize: float = 10,
            mean_sil_color: str = "red",
            legend_label: str = "Mean silhouette",
            legend_fontsize: float = 10,
            plot_mean_silhouette: bool = True):
        
        # Load samples, labels and calculate the silhouettes
        samples = self._data # normalized
        labels_clusters = self._samples_labels # indices of samples starting at 0
        silhouette_values = silhouette_samples(samples, labels_clusters)
        
        # For plotting, start with lower rank clusters in the bottom and go upward to highest ranks
        fig, ax = plt.subplots(figsize=figsize)
        y_lower = 10  # padding from bottom
        n_clusters = self.cmpf.shape[1]
        clusters_order = self.clusters_order
        cmap = plt.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, n_clusters))
        for rank in range(n_clusters, 0, -1):  # go from least to most important
            # Get the cluster index whose rank is 'rank'
            cluster_idx = np.where(clusters_order == rank)[0][0]
            
            # Get silhouette values for samples in this cluster
            ith_cluster_silhouette_values = silhouette_values[labels_clusters == cluster_idx]
            ith_cluster_silhouette_values.sort()
            
            size_cluster = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster
            
            color = colors[cluster_idx]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0, ith_cluster_silhouette_values,
                facecolor=color, edgecolor=color, alpha=0.7
            )

            # Plot the ranking and cluster label
            y_middle = y_lower + 0.5 * size_cluster
            if len(ith_cluster_silhouette_values) > 0:
                x_position = (ith_cluster_silhouette_values[-1]-min(0,np.min(silhouette_values)))/2
                if len(custom_labels) == 0:
                    ax.text(
                        x_position,
                        y_middle,
                        f'{rank}. Cluster #{cluster_idx+1}',
                        va='center',
                        ha='center',
                        fontsize=labels_fontsize
                    )
                else:
                    ax.text(
                        x_position,
                        y_middle,
                        f'{rank}. {custom_labels[cluster_idx]}',
                        va='center',
                        ha='center',
                        fontsize=labels_fontsize
                    )

            y_lower = y_upper + 10  # space between clusters

        ax.set_title(title, fontsize=text_fontsize)
        ax.set_xlabel(xlabel, fontsize=xy_labels_fontsize)
        ax.set_ylabel(ylabel, fontsize=xy_labels_fontsize)
        ax.set_yticks([])
        if plot_mean_silhouette:
            ax.axvline(x=np.mean(silhouette_values), color=mean_sil_color, linestyle="--", label=legend_label)
            ax.legend(loc='upper left', fontsize=legend_fontsize, bbox_to_anchor=legend_upper_left_bbox_anchor)
        plt.tight_layout()
        plt.show()

    def cmpf_silhouette_plot(
            self,
            figsize: tuple = (6,5),
            title: str = "CMPF-Silhouette Plot (ordered by cluster importance)",
            xlabel: str = "CMPF values",
            ylabel: str = "Samples (by cluster order)",
            colormap: str = "gist_rainbow",
            custom_labels: list = [],
            legend_upper_left_bbox_anchor: tuple = (1.02,0.7),
            text_fontsize: float = 14,
            labels_fontsize: float = 12,
            xy_labels_fontsize: float = 10,
            mean_sil_color: str = "red",
            legend_label: str = "Mean CMPF",
            legend_fontsize: float = 10,
            plot_mean_silhouette: bool = True):
        
        # Load samples, labels and calculate the silhouettes
        samples = self._data # normalized
        labels_clusters = self._samples_labels # indices of samples starting at 0
        silhouette_values = self.samples_scores['CMPF_value'].values
        
        # For plotting, start with lower rank clusters in the bottom and go upward to highest ranks
        fig, ax = plt.subplots(figsize=figsize)
        y_lower = 10  # padding from bottom
        n_clusters = self.cmpf.shape[1]
        clusters_order = self.clusters_order
        cmap = plt.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, n_clusters))
        for rank in range(n_clusters, 0, -1):  # go from least to most important
            # Get the cluster index whose rank is 'rank'
            cluster_idx = np.where(clusters_order == rank)[0][0]
            
            # Get silhouette values for samples in this cluster
            ith_cluster_silhouette_values = silhouette_values[labels_clusters == cluster_idx]
            ith_cluster_silhouette_values.sort()
            
            size_cluster = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster
            
            color = colors[cluster_idx]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0, ith_cluster_silhouette_values,
                facecolor=color, edgecolor=color, alpha=0.7
            )

            # Plot the ranking and cluster label
            y_middle = y_lower + 0.5 * size_cluster
            if len(ith_cluster_silhouette_values) > 0:
                x_position = (ith_cluster_silhouette_values[-1]-min(0,np.min(silhouette_values)))/2
                if len(custom_labels) == 0:
                    ax.text(
                        x_position,
                        y_middle,
                        f'{rank}. Cluster #{cluster_idx+1}',
                        va='center',
                        ha='center',
                        fontsize=labels_fontsize
                    )
                else:
                    ax.text(
                        x_position,
                        y_middle,
                        f'{rank}. {custom_labels[cluster_idx]}',
                        va='center',
                        ha='center',
                        fontsize=labels_fontsize
                    )

            y_lower = y_upper + 10  # space between clusters

        ax.set_title(title, fontsize=text_fontsize)
        ax.set_xlabel(xlabel, fontsize=xy_labels_fontsize)
        ax.set_ylabel(ylabel, fontsize=xy_labels_fontsize)
        ax.set_yticks([])
        if plot_mean_silhouette:
            ax.axvline(x=np.mean(silhouette_values), color=mean_sil_color, linestyle="--", label=legend_label)
            ax.legend(loc='upper left', fontsize=legend_fontsize, bbox_to_anchor=legend_upper_left_bbox_anchor)
        plt.tight_layout()
        plt.show()

    def plot_typical_samples_pca(
            self,
            normalization='var',
            random_state=None,
            figsize=(10,10),
            colormap='gist_rainbow',
            samples_size=30,
            highlight_samples='sc',
            typical_samples_linewidth=2,
            plot_labels=True,
            xy_plot_label=(0.1,0.1),
            label_fontsize=12,
            custom_labels=[],
            typical_sample_legend="Typical sample",
            cluster_legend="Cluster",
            legend_upper_left_bbox_anchor=(1.02,0.7),
            legend_ncols=1,
            legend_samples_size=10,
            legend_fontsize=12,
            legend_title="",
            title="PCA plot of typical samples",
            title_size=14):

        # Load denormalized data
        if self.som_object.missing:
            data_denorm = self.som_object.imput_missing(save=False) # pandas dataframe
            data_denorm = data_denorm.values
        else:
            data_denorm = self.som_object.denorm_data(self.som_object._data)
        # Normalize
        normalizer = NormalizerFactory.build(normalization)
        data = normalizer.normalize(data_denorm)

        # Load samples labels and clusters order
        samples_labels = self._samples_labels
        clusters_order = self.clusters_order

        # Fit the PCA and get variance explained
        pca = PCA(n_components=2, random_state=random_state)
        pca_data = pca.fit_transform(data)
        self.pca = pca
        var_pc1, var_pc2 = pca.explained_variance_ratio_

        # Plot samples, neurons and centroids
        f = plt.figure(figsize=figsize, dpi=300)
        plot_height = int(90*(var_pc1/var_pc2))
        gs_height = plot_height + 5
        gs = gridspec.GridSpec(gs_height, 100)
        ax1 = f.add_subplot(gs[:plot_height, 0:90])
        ax1.set_aspect('equal')
        # Create colormap
        cmap = plt.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, self.cmpf.shape[1]))
        # Plot samples
        ax1.scatter(
            pca_data[:, 0], pca_data[:, 1],
            c=colors[samples_labels],
            edgecolors='k',
            s=samples_size,
            marker='o',
            zorder=1)

        # Plot typical samples
        if highlight_samples == 'cmpf-sc':
            typical_samples = self.cmpf_sc_typical_samples_indices
        elif highlight_samples == 'cmpf-dist':
            typical_samples = self.cmpf_dists_typical_samples_indices
        elif highlight_samples == 'sc':
            typical_samples = self.sc_typical_samples_indices
        else:
            typical_samples = highlight_samples
        for typ_idx, typical in enumerate(typical_samples):
            ax1.scatter(
                pca_data[typical, 0], pca_data[typical, 1],
                edgecolors='k',
                facecolors='none',
                s=samples_size,
                marker='o',
                linewidths=typical_samples_linewidth,
                zorder=2
            )

            # Plot typical samples names
            if plot_labels:
                ax1.text(
                    pca_data[typical,0]+xy_plot_label[0],
                    pca_data[typical,1]+xy_plot_label[1],
                    custom_labels[typ_idx] if len(custom_labels) != 0 else self.som_object._sample_names[typical],
                    fontsize=label_fontsize,
                    # bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
                    zorder=3)

        # Plot legend
        custom_legend = []
        labels = []
        typical_samples_handle = mlines.Line2D([], [], color='none', marker='o', linestyle='None', mew=typical_samples_linewidth, mec='k', markersize=legend_samples_size)
        custom_legend.append(typical_samples_handle)
        labels.append(typical_sample_legend)
        # Ranking handles
        clusters_order = self.clusters_order
        for rank_idx in range(len(clusters_order)):
            clust_idx = np.where(clusters_order == rank_idx+1)[0][0]
            sample_cluster_handle = mlines.Line2D([], [], color=colors[clust_idx], marker='o', linestyle='None', mec='k', markersize=legend_samples_size)
            custom_legend.append(sample_cluster_handle)
            labels.append(f'{rank_idx+1}. {cluster_legend} #{clust_idx+1}')

        # Plot configs
        ax1.legend(handles=custom_legend, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='upper left', bbox_to_anchor=legend_upper_left_bbox_anchor, ncols=legend_ncols, fontsize=legend_fontsize, title=legend_title)

        ax1.set_xlabel(f"PC 1 ({round(100*var_pc1, 2)}%)")
        ax1.set_ylabel(f"PC 2 ({round(100*var_pc2, 2)}%)")
        ax1.set_title(title, fontsize=title_size)

        # ADD WATERMARK
        # Add white space subplot below the plot
        # image width is 4 times its height
        image_width = 4*(gs_height-plot_height)
        ax2 = f.add_subplot(gs[plot_height:gs_height, 0:image_width], zorder=-1)
        ax2.imshow(self.foot, aspect='equal', alpha=1)
        ax2.axis('off')

        plt.show()

    def representatives_pca_visualization(
                                          self,
                                          figsize=(10,10),
                                          normalization='var',
                                          random_state=None,
                                          show_samples=True,
                                          samples_labels_colors=True,
                                          samples_size=30,
                                          samples_color='white',
                                          samples_edgecolor='black',
                                          samples_marker='o',
                                          samples_legend='Samples',
                                          show_neurons=True,
                                          colormap='gist_rainbow',
                                          alpha_neurons=0.5,
                                          neurons_max_size=60,
                                          neurons_marker='h',
                                          watermark_neurons=False,
                                          y_watermark_neurons_text=-0.03,
                                          watermark_neurons_fontsize=5,
                                          show_centroids=True,
                                          centroids_size=60,
                                          centroids_edgecolor='black',
                                          centroids_marker='P',
                                          clusters_legend='Cluster',
                                          title='Samples, Neurons and Centroids',
                                          title_size=12,
                                          legend_upper_left_bbox_anchor=(1.02,0.7),
                                          legend_ncols=1,
                                          sample_fontsize=6,
                                          neuron_fontsize=12,
                                          centroid_fontsize=8,
                                          legend_fontsize=10,
                                          plot_typical_samples=True,
                                          typical_samples_legend="Typical samples",
                                          highlight_samples='cmpf-sc',
                                          highlight_samples_fontsize=8,
                                          typical_samples_linewidth=2.0,
                                          plot_ranking=True,
                                          plot_typical_labels=True,
                                          xy_typical_label=(0.1,0.1),
                                          xy_rank=(0.5,0.1),
                                          legend_title=""
                                          ):

        # Load denormalized data
        if self.som_object.missing:
            data_denorm = self.som_object.imput_missing(save=False) # pandas dataframe
            data_denorm = data_denorm.values
        else:
            data_denorm = self.som_object.denorm_data(self.som_object._data)
        # Normalize
        normalizer = NormalizerFactory.build(normalization)
        data = normalizer.normalize(data_denorm)

        # Load normalized neuron weights matrix
        neurons = self._neurons

        # Load K-means neurons labels, samples labels, normalized centroids, max CMPF values of each neuron and clusters ranking
        neurons_labels = self._neurons_labels
        samples_labels = self._samples_labels
        max_clust = np.max(neurons_labels)
        clusters_centroids = self._clusters_centroids # 
        max_cmpf = np.max(self.cmpf, axis=1)
        # Get the clusters ordering
        clusters_order = self.clusters_order

        # Fit the PCA and project neurons weights
        pca = PCA(n_components=2, random_state=random_state)
        pca_data = pca.fit_transform(data)
        self.pca = pca
        pca_neurons = pca.transform(neurons)
        pca_centroids = pca.transform(clusters_centroids)
        var_pc1, var_pc2 = pca.explained_variance_ratio_

        # Plot samples, neurons and centroids
        f = plt.figure(figsize=figsize, dpi=300)
        plot_height = int(90*(var_pc1/var_pc2))
        gs_height = plot_height + 5
        gs = gridspec.GridSpec(gs_height, 100)
        ax1 = f.add_subplot(gs[:plot_height, 0:90])
        ax1.set_aspect('equal')
        # Create colormap
        cmap = cm.get_cmap(colormap)
        norm = mpl.colors.Normalize(vmin=np.nanmin(neurons_labels), vmax=np.nanmax(neurons_labels))
        # Plot samples
        if show_samples:
            ax1.scatter(
                pca_data[:, 0], pca_data[:, 1],
                c=cmap(norm(samples_labels)) if samples_labels_colors else samples_color,
                edgecolors=samples_edgecolor,
                s=samples_size,
                marker=samples_marker,
                zorder=1,
                label='Samples'
            )

        # Plot neurons
        if show_neurons:
            ax1.scatter(
                pca_neurons[:, 0], pca_neurons[:, 1],
                c=neurons_labels,
                cmap=cmap,
                norm=norm,
                alpha=max_cmpf,
                edgecolors=cmap(norm(neurons_labels)),
                s=neurons_max_size*max_cmpf,
                marker=neurons_marker,
                zorder=2,
                label='Neurons'
            )
        if watermark_neurons:
            neurons_labels = self._neurons_labels-1
            mean_cmpf = np.mean(self.cmpf, axis=0)
            mean_cmpf_per_neuron = mean_cmpf[neurons_labels]
            for idx, (x, y) in enumerate(pca_neurons):
                if max_cmpf[idx] >= mean_cmpf_per_neuron[idx]:
                    ax1.text(x, y+y_watermark_neurons_text, str(idx + 1), fontsize=watermark_neurons_fontsize, ha='center', va='bottom', color='black')
        # Plot centroids
        if show_centroids:
            ax1.scatter(
                pca_centroids[:, 0], pca_centroids[:, 1],
                c=list(range(1,max_clust+1)),
                cmap=cmap,
                norm=norm,
                edgecolors=centroids_edgecolor,
                s=centroids_size,
                marker=centroids_marker,
                zorder=3,
                label='Centroids'
            )
        # Plot typical samples
        if plot_typical_samples:
            if highlight_samples == 'cmpf-sc':
                typical_samples = self.cmpf_sc_typical_samples_indices
            elif highlight_samples == 'cmpf-dist':
                typical_samples = self.cmpf_dists_typical_samples_indices
            elif highlight_samples == 'sc':
                typical_samples = self.sc_typical_samples_indices
            else:
                typical_samples = highlight_samples
            for typical in typical_samples:
                typical_scores = self.samples_scores[self.samples_scores['Sample_index'] == typical]
                c_label = typical_scores['Cluster']
                ax1.scatter(
                    pca_data[typical, 0], pca_data[typical, 1],
                    c=c_label,
                    cmap=cmap,
                    norm=norm,
                    edgecolors=samples_edgecolor,
                    facecolors='none' if samples_labels_colors else None,
                    s=samples_size,
                    marker=samples_marker,
                    linewidths=typical_samples_linewidth,
                    zorder=4,
                    label='Typical samples'
                )
                if plot_typical_labels:
                    ax1.text(
                        pca_data[typical,0]+xy_typical_label[0],
                        pca_data[typical,1]+xy_typical_label[1],
                        self.som_object._sample_names[typical],
                        fontsize=highlight_samples_fontsize,
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
                        zorder=4)
                if plot_ranking:
                    ax1.text(
                        pca_data[typical,0]-xy_rank[0],
                        pca_data[typical,1]-xy_rank[1],
                        f'Rank {clusters_order[c_label-1]}',
                        c=cmap(norm(c_label)),
                        fontsize=highlight_samples_fontsize,
                        weight='bold',
                        zorder=4,
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
        # Plot parameters
        ax1.set_xlabel(f"PC 1 ({round(100*var_pc1, 2)}%)")
        ax1.set_ylabel(f"PC 2 ({round(100*var_pc2, 2)}%)")
        ax1.set_title(title, fontsize=title_size)

        # Plot legend
        # Build the samples ranking
        cmap = plt.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, len(clusters_order)))
        custom_legend = []
        labels = []
        if samples_labels_colors or plot_typical_samples:
            # Typical sample handles
            if plot_typical_samples:
                typical_samples_handle = mlines.Line2D([], [], color='none', marker=samples_marker, linestyle='None', mew=typical_samples_linewidth, mec=samples_edgecolor, markersize=sample_fontsize)
                custom_legend.append(typical_samples_handle)
                labels.append(typical_samples_legend)
            # Ranking handles
            clusters_order = self.clusters_order
            for rank_idx in range(len(clusters_order)):
                clust_idx = np.where(clusters_order == rank_idx+1)[0][0]
                sample_cluster_handle = mlines.Line2D([], [], color=colors[clust_idx], marker=samples_marker, linestyle='None', mec=samples_edgecolor, markersize=sample_fontsize)
                custom_legend.append(sample_cluster_handle)
                labels.append(f'{rank_idx+1}. {clusters_legend} #{clust_idx+1}')

        else:
            if show_samples:
                samples_handle = mlines.Line2D([], [], color=samples_color, marker=samples_marker, linestyle='None', mec=samples_edgecolor, markersize=sample_fontsize)
                custom_legend.append(samples_handle)
                labels.append(samples_legend)
            if show_neurons:
                for clust_idx in range(1,max_clust+1):
                    neuron_cluster_handle = mlines.Line2D([], [], color=cmap(norm(clust_idx)), alpha=alpha_neurons, marker=neurons_marker, linestyle='None', mec=cmap(norm(clust_idx)), markersize=neuron_fontsize)
                    centroid_cluster_handle = mlines.Line2D([], [], color=cmap(norm(clust_idx)), marker=centroids_marker, linestyle='None', mec=centroids_edgecolor, markersize=centroid_fontsize)
                    typical_samples_handle = mlines.Line2D([], [], color=cmap(norm(clust_idx)), marker=samples_marker, linestyle='None', mec=samples_edgecolor, mew=typical_samples_linewidth, markersize=sample_fontsize)
                    custom_legend.append((neuron_cluster_handle, centroid_cluster_handle, typical_samples_handle))
                    labels.append(f'{clusters_legend} #{clust_idx}')

        ax1.legend(handles=custom_legend, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='upper left', bbox_to_anchor=legend_upper_left_bbox_anchor, ncols=legend_ncols, fontsize=legend_fontsize, title=legend_title)

        # ADD WATERMARK
        # Add white space subplot below the plot
        # image width is 4 times its height
        image_width = 4*(gs_height-plot_height)
        ax2 = f.add_subplot(gs[plot_height:gs_height, 0:image_width], zorder=-1)
        ax2.imshow(self.foot, aspect='equal', alpha=1)
        ax2.axis('off')

        plt.show()

    # @staticmethod
    # def proto_cluster_energies(sample, cluster, num_permutations=1000, seed=None):
    #     """
    #     This function evaluates the energies of each SOM proto-cluster.
        
    #     In the case where neurons are approximadetly equidistant of a cluster centroid and, at the same time, have an even number of hits, the CMPF won't be enough to distinguish which neuron is more typical, relying on decimals to elect the typical neuron.
        
    #     Given that, an energy function based on the Energy Distance Hypothesis Test (E-test) can be used to measure the mean distance between the selected sample (proto-cluster of samples associated to the winning neuron or BMU) and its respective cluster or population.
        
    #     In this sense, proto-clusters with lower energies can be considered more typical.

    #     Parameters:
    #         sample (np.ndarray or pd.DataFrame): sample to be tested (e.g., 1 or few samples)
    #         cluster (np.ndarray or pd.DataFrame): population or group from which the sample was drawn
    #         num_permutations (int): number of permutations for p-value estimation
    #         seed (int): random seed for reproducibility
        
    #     Returns:
    #         energy (float): observed energy statistic
    #     """

    #     if seed is not None:
    #         np.random.seed(seed)

    #     sample = np.asarray(sample)
    #     sample_and_population = np.asarray(cluster)
    #     n_total = sample_and_population.shape[0]
        
    #     if sample.ndim == 1:
    #         sample = sample[np.newaxis, :]

    #     # Separate the sample from the cluster array
    #     sample_struct = sample.view([('', sample.dtype)] * sample.shape[1])
    #     population_struct = sample_and_population.view([('', sample_and_population.dtype)] * sample_and_population.shape[1])
    #     mask = ~np.isin(population_struct.ravel(), sample_struct.ravel())
    #     population = sample_and_population[mask]

    #     n_sample, n_population = sample.shape[0], population.shape[0]

    #     # Distances
    #     dists = np.linalg.norm(sample[:, np.newaxis, :] - population, axis=2)
    #     energy = np.sum(dists) / (n_sample * n_population)

    #     # Do the permutation test to calculate the p_value
    #     count = 0
    #     for _ in range(num_permutations):
    #         idx = np.random.permutation(n_total)
    #         sample_perm = sample_and_population[idx[:n_sample]]
    #         population_perm = sample_and_population[idx[n_sample:]]

    #         dist_perm = np.linalg.norm(sample_perm[:, np.newaxis, :] - population_perm, axis=2)

    #         energy_perm = np.sum(dist_perm) * 2 / (n_sample * n_population)

    #         if energy_perm >= energy:
    #             count += 1

    #     p_value = (count + 1) / (num_permutations + 1) # +1: corrected p-value

    #     return energy, p_value

    # def calculate_neurons_hypothesis_test(self, num_permutations=1000, seed=None):
    #     """
    #     This function was designed to assess neurons with different 
    #     This function can be used to assess if a set of samples associated to a given BMU is more typical than another BMU with a different set of samples.
    #     """

    #     data = self._data # normalized
    #     samples_bmu_label = self.samples_bmu.values-1 # indices start at 0 (n_samples)
    #     neurons_clusters_labels = self._neurons_labels # indices start at 1 (n_neurons)
    #     samples_clusters_labels = self._samples_labels # indices start at 0 (n_samples)

    #     # Now calculate the E_stats and p_values for each BMU
    #     E_stats = np.full(self._neurons.shape[0], np.nan)
    #     p_values = np.full(self._neurons.shape[0], np.nan)
    #     for bmu_idx in np.unique(samples_bmu_label):
    #         sample = data[samples_bmu_label==bmu_idx] # the samples associated to the BMU
    #         clust_idx = neurons_clusters_labels[bmu_idx]-1 # the cluster of the BMU
    #         cluster = data[samples_clusters_labels==clust_idx] # all the samples in the cluster of the BMU
    #         E_stat, p_value = self.multivariate_hypothesis_test(sample, cluster, num_permutations, seed=seed)
    #         E_stats[bmu_idx] = E_stat
    #         p_values[bmu_idx] = p_value

    #     self.E_stats = E_stats
    #     self.p_values = p_values

    #     return E_stats, p_values

    def generate_hex_lattice(self, n_columns, n_rows):
        """
        Generates the xy coordinates of the neurons for an odd-r hexagonal grid.
        Args:
            n_rows: number of rows in the Kohonen map.
            n_columns: number of columns in the Kohonen map.

        Returns:
            Coordinates of the [x,y] format for the neurons in a hexagonal grid.
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

    # def representatives_pca_visualization(self,
    #                                       colormap="gist_rainbow",
    #                                       show_samples=True,
    #                                       samples_size=30,
    #                                       samples_color='white',
    #                                       samples_edgecolor='black',
    #                                       samples_marker='o',
    #                                       samples_legend='Samples',
    #                                       show_neurons=True,
    #                                       neurons_marker='h',
    #                                       alpha_neurons=0.5,
    #                                       neurons_max_size=100,
    #                                       show_centroids=True,
    #                                       centroids_marker='X',
    #                                       centroids_edgecolor='black',
    #                                       clusters_legend='Cluster',
    #                                       centroids_size=100,
    #                                       typical_samples_size=100,
    #                                       watermark_neurons=False,
    #                                       y_watermark_neurons_text=0.,
    #                                       watermark_neurons_fontsize=5,
    #                                       legend_upper_left_bbox_anchor=(1.02,0.7),
    #                                       xlabel='PC1',
    #                                       ylabel='PC2'):
    #     """
    #     Visualize the samples, SOM neurons and K-Means centroids in a PCA projection. The representatives are considered to be the clusters centroids and the typical neurons and samples. The non-typical neurons were assigned a transparency according to its maximum responsibility.

    #     Args:
    #         colormap (str): colormap used for the centroids and typical neurons/samples.
    #     """

    #     _data = self._data
    #     _neurons = self._neurons
    #     _clusters_centroids = self._clusters_centroids
    #     _neurons_labels = self._neurons_labels
    #     cmpf = self.cmpf
    #     typical_neurons = self.typical_neurons
    #     typical_samples = self.typical_samples
    #     max_cmpf = np.max(cmpf, axis=1)

    #     # Fit the PCA on the dataset
    #     pca = PCA(n_components=2)
    #     data_2d = pca.fit_transform(_data)
    #     neurons_2d = pca.transform(_neurons)
    #     centroids_2d = pca.transform(_clusters_centroids)

    #     # Plot the projected neurons with samples and centroids
    #     plt.figure(figsize=(10, 8))
    #     if show_samples:
    #         samples_scatter = plt.scatter(
    #             data_2d[:, 0], data_2d[:, 1],
    #             c=samples_color,
    #             edgecolors=samples_edgecolor,
    #             s=samples_size,
    #             marker=samples_marker,
    #             label=samples_legend
    #         )
    #     if show_neurons:
    #         neurons_scatter = plt.scatter(
    #             neurons_2d[:, 0], neurons_2d[:, 1],
    #             c=_neurons_labels,
    #             s=neurons_max_size*max_cmpf,
    #             edgecolors='black',
    #             cmap=colormap,
    #             alpha=max_cmpf,
    #             marker=neurons_marker
    #         )
    #     if show_centroids:
    #         centroids_scatter = plt.scatter(
    #             centroids_2d[:, 0], centroids_2d[:, 1], 
    #             c=list(range(1, np.max(_neurons_labels) + 1)),
    #             edgecolors=centroids_edgecolor,
    #             s=centroids_size,
    #             marker=centroids_marker,
    #             linewidths=2,
    #             cmap=colormap,
    #             label='Centroids'
    #         )
    #     typical_neurons_scatter = plt.scatter(
    #         neurons_2d[typical_neurons-1, 0], neurons_2d[typical_neurons-1, 1],
    #         c=list(range(1, np.max(_neurons_labels) + 1)),
    #         s=neurons_max_size*np.max(cmpf, axis=0),
    #         edgecolors='black',
    #         cmap=colormap,
    #         marker=neurons_marker,
    #         linewidths=2,
    #         label='Typical neurons'
    #     )
    #     typical_samples_scatter = plt.scatter(
    #         data_2d[typical_samples-1, 0], data_2d[typical_samples-1, 1],
    #         c=list(range(1, np.max(_neurons_labels) + 1)),
    #         edgecolors=samples_edgecolor,
    #         s=typical_samples_size,
    #         marker=samples_marker,
    #         linewidths=2,
    #         cmap=colormap,
    #         label='Typical samples'
    #     )
    #     if watermark_neurons:
    #         for idx, (x, y) in enumerate(neurons_2d):
    #             plt.text(x, y+y_watermark_neurons_text, str(idx+1), fontsize=watermark_neurons_fontsize, ha='center', va='center', color='black')

    #     # Legend
    #     custom_legend = []
    #     labels = []
    #     if show_samples:
    #         samples_handle = mlines.Line2D([], [], color=samples_color, marker=samples_marker, linestyle='None', mec=samples_edgecolor)
    #         custom_legend.append(samples_handle)
    #         labels.append(samples_legend)
    #     if show_neurons:
    #         max_clust=np.max(_neurons_labels)
    #         cmap=cm.get_cmap(colormap)
    #         norm = mpl.colors.Normalize(vmin=np.nanmin(_neurons_labels), vmax=np.nanmax(_neurons_labels))
    #         for clust_idx in range(1,max_clust+1):
    #             neuron_cluster_handle = mlines.Line2D([], [], color=cmap(norm(clust_idx)), alpha=alpha_neurons, marker=neurons_marker, linestyle='None', mec=cmap(norm(clust_idx)))
    #             centroid_cluster_handle = mlines.Line2D([], [], color=cmap(norm(clust_idx)), marker=centroids_marker, linestyle='None', mec=centroids_edgecolor)
    #             custom_legend.append((neuron_cluster_handle, centroid_cluster_handle))
    #             labels.append(f'{clusters_legend} {clust_idx}')

    #     plt.legend(handles=custom_legend, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='upper left', bbox_to_anchor=legend_upper_left_bbox_anchor)
    #     plt.title('SOM + K-Means centroids, typical neurons and typical samples\nprojected with PCA')
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.show()

    # def typical_samples_multivariate_hypothesis_test(self,
    #                                                  num_permutations=1000,
    #                                                  clusters_to_test=[],
    #                                                  p=0.05,
    #                                                  seed=None):
    #     """
    #     Plot the results of the multivariate hypothesis test for a given set of clusters and its samples considered as typical.
        
    #     Parameters:
    #         clusters_to_test (list): list with the indices of the clusters that will be tested (starting at 1). If empty, all clusters will be tested.
    #     """

    #     data = self._data # normalized
    #     typical_samples = self.typical_samples-1 # indices start at 0
    #     samples_clusters_labels = self._samples_labels # ranging from 0 to n_clusters-1

    #     E_stats = []
    #     p_values = []
    #     clusters = []

    #     if clusters_to_test == []:
    #         clusters_to_test = np.unique(samples_clusters_labels)
    #     else:
    #         clusters_to_test = [i - 1 for i in clusters_to_test]  # user provides 1-based indices

    #     for clust_idx in clusters_to_test:
    #         cluster = data[samples_clusters_labels == clust_idx]
    #         sample = data[typical_samples[clust_idx]]
    #         E_stat, p_value = self.multivariate_hypothesis_test(sample, cluster, num_permutations, seed=seed)

    #         E_stats.append(E_stat)
    #         p_values.append(p_value)
    #         clusters.append(clust_idx + 1)  # back to 1-based for plotting

    #     # Plotting
    #     fig, ax1 = plt.subplots(figsize=(10, 6))

    #     color = 'tab:blue'
    #     ax1.set_xlabel('Cluster')
    #     ax1.set_ylabel('E-statistic', color=color)
    #     ax1.plot(clusters, E_stats, marker='o', color=color, label='E-statistic')
    #     ax1.tick_params(axis='y', labelcolor=color)
    #     ax1.set_xticks(clusters)

    #     # Second axis for p-values
    #     ax2 = ax1.twinx()

    #     color = 'tab:red'
    #     ax2.set_ylabel('p-value', color=color)
    #     ax2.plot(clusters, p_values, marker='x', linestyle='--', color=color, label='p-value')
    #     ax2.tick_params(axis='y', labelcolor=color)

    #     # Reference line for significance threshold (e.g., p=0.05)
    #     ax2.axhline(y=p, color='red', linestyle=':', label=f'Significance threshold ({p})')

    #     # Title and grid
    #     plt.title('Multivariate Hypothesis Test per Cluster')
    #     fig.tight_layout()
    #     plt.grid(True)
    #     plt.show()