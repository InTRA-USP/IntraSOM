from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from numpy import nanmean, dot
import matplotlib as mpl
from matplotlib.patches import RegularPolygon
from matplotlib import cm
import os
from math import sqrt
from sklearn.preprocessing import minmax_scale
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union
import geopandas as gpd
import pkg_resources
from PIL import Image


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
        # Load foot image
        image_file = pkg_resources.resource_filename('intrasom', 'images/foot.jpg')
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
    
    def results_cluster(self, clusters, save=True, savetype="parquet"):
        """
        Returns a DataFrame containing the results of grouping the data, with the addition of the assigned cluster labels.

        Args:
            clusters (numpy.ndarray): A one-dimensional array containing the cluster labels assigned to each data point.
            save (bool, optional): Indicates whether the results should be saved in an Excel file. The default is True.

        Returns:
            pandas.DataFrame: A DataFrame containing the results of clustering the data, including the assigned cluster labels.
        """

        df = self.neurons_dataframe.copy()
        df[f"{clusters.max()}_clusters"] = clusters.flatten()
        df = df.iloc[self.bmus,:]

        df.set_index(self.sample_names, inplace=True)

        if save:
            path = 'Results'
            os.makedirs(path, exist_ok=True)
            if savetype == "xlsx":
                df.to_excel(f"Results/{self.name}_results_{clusters.max()}_clusters.xlsx")
            elif savetype == "parquet":
                df.to_parquet(f"Results/{self.name}_results_{clusters.max()}_clusters.parquet")

        return df
            
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
                                         facecolor= cm.jet(norm(umat[j][i])),
                                         alpha=alfa, 
                                         zorder=0)#, edgecolor='black')

                    ax.add_patch(hexagon)

                    # Right Hexagon
                    if not np.isnan(um[j, i, 0]):
                        hexagon = RegularPolygon((xx[(j, i)]*2+1,
                                              yy[(j,i)]*2),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,0])),
                                             alpha=alfa, 
                                             zorder=0)
                        ax.add_patch(hexagon)

                    # Upper Right Hexagon
                    if not np.isnan(um[j, i, 1]):
                        hexagon = RegularPolygon((xx[(j, i)]*2+0.5,
                                              yy[(j,i)]*2+(np.sqrt(3)/2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,1])),
                                             alpha=alfa, 
                                             zorder=0)
                        ax.add_patch(hexagon)

                    # Upper Left Hexagon
                    if not np.isnan(um[j, i, 2]):
                        hexagon = RegularPolygon((xx[(j, i)]*2-0.5,
                                              yy[(j,i)]*2+(np.sqrt(3)/2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,2])),
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

        cmap = cm.get_cmap(colormap)
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
                dissolved_geometry = cascaded_union(geo_series)
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
    

    def build_umatrix(self, expanded=False, log=False):
        """
        Function to calculate the U Matrix of unified distances from the
        trained weight matrix.

        Args:
            expanded: boolean value to indicate whether the return will be from the summarized
                or unified matrix of distances (average of distances from the 6
                neighborhood BMUs) or expanded (all distance values)
                
        Returns:
            Expanded or summarized unified distance matrix.
        """
        # Function to find distance quickly
        def fast_norm(x):
            """
            Returns the L2 norm of a 1-D array.
            """
            return sqrt(dot(x, x.T))

        # Matrix of BMUs weights
        weights = np.reshape(self.codebook, (self.mapsize[1], self.mapsize[0], self.codebook.shape[1]))

        # Neighbor hexagonal search
        ii = [[1, 1, 0, -1, 0, 1], [1, 0,-1, -1, -1, 0]]
        jj = [[0, 1, 1, 0, -1, -1], [0, 1, 1, 0, -1, -1]]

        # Initialize U Matrix
        um = np.nan * np.zeros((weights.shape[0], weights.shape[1], 6))

        # Fill U Matrix
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                w_2 = weights[y, x]
                e = y % 2 == 0
                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                    if (x+i >= 0 and x+i < weights.shape[1] and y+j >= 0 and y+j < weights.shape[0]):
                        w_1 = weights[y+j, x+i]
                        um[y, x, k] = fast_norm(w_2-w_1)
        if expanded:
            # Expanded U matrix
            return np.log(um) if log else um
        else:
            # Reduced U matrix
            return nanmean(np.log(um), axis=2) if log else nanmean(um, axis=2)
        

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