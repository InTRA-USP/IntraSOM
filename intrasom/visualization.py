import numpy as np
from sklearn.preprocessing import minmax_scale
from numpy import nan, dot, nanmean
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from tqdm.auto import tqdm
from scipy import sparse as sp
from PIL import Image
import glob
import os
from textwrap import fill
from numpy import pi, sin, cos
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import pkg_resources
import plotly.graph_objs as go
from scipy.ndimage import rotate
from skimage.transform import resize
from sklearn.metrics.pairwise import nan_euclidean_distances
import statistics



class PlotFactory(object):

    def __init__(self, som_object):
        self.name = som_object.name
        self.codebook = som_object.codebook.matrix
        self.mapsize = som_object.mapsize
        self.bmus = som_object._bmu[0].astype(int)
        self.neuron_matrix = som_object.neuron_matrix
        self.component_names = som_object._component_names
        self.sample_names = som_object._sample_names
        self.unit_names = som_object._unit_names
        self.rep_sample = som_object.rep_sample
        self.data_denorm = som_object.denorm_data(som_object._data)
        self.data_proj_norm = som_object.data_proj_norm
        
        # Load foot image
        image_file = pkg_resources.resource_filename('intrasom', 'images/foot.jpg')
        self.foot = Image.open(image_file)

    def build_umatrix(self, expanded=False, log=False):
        """
        Function to calculate the U-Matrix of unified distances from the trained weight matrix.

        Args:
            exapanded: Boolean value to indicate whether the return will be the summarized 
                U-Matrix (average distances of the 6 neighboring BMUs) or the expanded 
                U-Matrix (all distance values).
            
            log: Returns the base 10 logarithm for the distance values. It is used when 
                there are samples with a large dissimilarity boundary that masks the 
                visualization of the U-Matrix. The logarithmic transformation of these 
                values allows for a better visualization of the matrix.

        Returns:
            Expanded or summarized U-Matrix of distances.
        """
        # Function to find distance quickly
        def fast_norm(x):
            """
            Returns the L2 norm of a 1-D array.
            """
            return sqrt(dot(x, x.T))

        # Neurons weights matrix
        weights = np.reshape(self.codebook, (self.mapsize[1], self.mapsize[0], self.codebook.shape[1]))

        # Neighbor hexagonal search
        ii = [[1, 1, 0, -1, 0, 1], [1, 0,-1, -1, -1, 0]]
        jj = [[0, 1, 1, 0, -1, -1], [0, 1, 1, 0, -1, -1]]

        # Initialize U Matrix
        um = np.nan * np.zeros((weights.shape[0], weights.shape[1], 6))

        # Fill U-matrix
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                w_2 = weights[y, x]
                e = y % 2 == 0
                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                    if (x+i >= 0 and x+i < weights.shape[1] and y+j >= 0 and y+j < weights.shape[0]):
                        w_1 = weights[y+j, x+i]
                        um[y, x, k] = fast_norm(w_2-w_1)
        if expanded:
            # Expanded U Matrix
            return np.log(um) if log else um
        else:
            # Reduced U Matrix
            return nanmean(np.log(um), axis=2) if log else nanmean(um, axis=2)
                        
    def plot_umatrix(self,
                     figsize = (10,10),
                     hits = True,
                     title = "U-Matrix",
                     title_size = 40,
                     title_pad = 25,
                     legend_title = "Distance",
                     legend_title_size = 25,
                     legend_ticks_size = 20,
                     save = True,
                     watermark_neurons = False,
                     watermark_neurons_alfa = 0.5,
                     neurons_fontsize = 7,
                     file_name = None,
                     file_path = False, 
                     resume = False,
                     label_plot = False, 
                     label_plot_name = None,
                     project_samples_label = None,
                     samples_label = False,
                     samples_label_index = None,
                     samples_label_fontsize = 8,
                     save_labels_rep = False,
                     label_title_xy = (0,0.5),
                     log=False):
        """
        Function to plot the U-Matrix of unified distances.

        Args:
            figsize:  Size of the U-Matrix plotting area. The adjustment of these values is 
                crucial for the proper distribution of the plotted objects and strongly depends 
                on the shape of the trained map (number of rows and columns). Default: (10, 10)

            hits: Boolean value to indicate whether to plot the hits of input vectors on the 
                BMUs (proportional to the number of vectors per BMU). These hits are visualized as 
                white hexagons with size proportional to the number of input samples represented 
                by that BMU.

            title: Title of the created figure. Default: "U-Matrix"

            title_size: Size of the plotted title. Default: 40

            title_pad: Spacing between the title and the top of the matrix. Default: 25

            legend_title: Title of the legend color bar. Default: "Distance"

            legend_title_size: Size of the legend title. Default: 25

            legend_ticks_size: Size of the digits on the legend color bar. Default: 20

            save: Boolean value to define whether to save the created image. The image will be saved in the 
                directory (Plotagens/Matriz_U). Default: True

            watermark_neurons: Boolean value to add a watermark with the neuron numbers to the image. 
                Default: False

            watermark_neurons_alfa: Value between 0 and 1 indicating the transparency of the plotted 
                neuron template over the U-Matrix. The closer to 1, the lower the transparency. Default: 0.5.

            neurons_fontsize: Font size for the neuron numbers to be plotted. Default: 7.

            file_name: Name to be given to the saved file. If no name is provided, the project name 
                will be used.

            file_path: System path where the image should be saved, if a custom path is preferred.

            resume: Boolean value to plot only the upper part of the U-Matrix, omitting the mirrored 
                lower part. Default: False

            label_plot: Boolean value to add labels to the hexagons of the U-Matrix according to a 
                boolean variable present in the training. Default: False

            label_plot_names: Name of the variable for plotting. The presence of the variable is 
                indicated by white hits, and the absence by black hits. Default: None

            samples_label: Boolean value to indicate the plotting of labels for selected samples. 
                Default: False.

            samples_label_index: List of indices of the selected samples for label plotting. 
                Default: None. "All" for plotting all samples.

            samples_label_fontsize: Font size for the plotted labels. Default: 8.

            save_labels_rep: Boolean value to indicate whether to save a .txt file in Results with 
                the selected samples and their representativeness order in each BMU. Default: False.

            label_title_xy: Coordinates (x, y) to position the label title. Default: (-0.02, 1.1)

            log: Boolean value to plot the U-Matrix on a logarithmic scale for better visualization 
                of dissimilarity boundaries in the presence of outliers.

        Returns:
            The image with the plot of the U-Matrix of unified distances.
        """
        def select_keys(original_dict, keys_list):
            new_dict = {}
            keys_list = np.array(keys_list)
            if keys_list.shape[0]>1:
                for key in keys_list:
                    if key in original_dict:
                        new_dict[key] = original_dict[key]
            else:
                if keys_list[0] in original_dict:
                    new_dict[keys_list[0]] = original_dict[keys_list[0]]
            return new_dict
        
        def search_strings(search_list, target_list):
            found_index = None
            found_string = None

            for search_string in search_list:
                for index, target_string in enumerate(target_list):
                    if search_string == target_string:
                        if found_index is None or index < found_index:
                            found_index = index
                            found_string = search_string
                        break

            return found_string, found_index

        if file_name is None:
            file_name = f"U_Matrix_{self.name}"

        if hits:
            bmu_dic = self.hits_dictionary

        # Create coordinates
        xx = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,0], (self.mapsize[1], self.mapsize[0]))
        yy = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,1], (self.mapsize[1], self.mapsize[0]))

        # U Matrix
        um = self.build_umatrix(expanded = True, log=log)
        umat = self.build_umatrix(expanded = False, log=log)
        
        
        if resume:
            # Plotting
            prop = self.mapsize[1]*0.8660254/self.mapsize[0]
            f = plt.figure(figsize=(5, 5*prop), dpi=300)
            f.patch.set_facecolor('blue')
            ax = f.add_subplot()
            ax.set_aspect('equal')

            # Normalize colors for all hexagons
            norm = mpl.colors.Normalize(vmin=np.nanmin(um), vmax=np.nanmax(um))
            counter = 0

            for j in range(self.mapsize[1]):
                for i in range(self.mapsize[0]):
                    # Central Hexagon
                    hex = RegularPolygon((xx[(j, i)]*2,
                                          yy[(j,i)]*2),
                                         numVertices=6,
                                         radius=1/np.sqrt(3),
                                         facecolor= cm.jet(norm(umat[j][i])),
                                         alpha=1)#, edgecolor='black')

                    ax.add_patch(hex)

                    # Upper Right Hexagon
                    if not np.isnan(um[j, i, 0]):
                        hex = RegularPolygon((xx[(j, i)]*2+1,
                                              yy[(j,i)]*2),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,0])),
                                             alpha=1)
                        ax.add_patch(hex)

                    # Upper Left Hexagon
                    if not np.isnan(um[j, i, 1]):
                        hex = RegularPolygon((xx[(j, i)]*2+0.5,
                                              yy[(j,i)]*2+(np.sqrt(3)/2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,1])),
                                             alpha=1)
                        ax.add_patch(hex)

                    # Hexagono Superior Esquerdo
                    if not np.isnan(um[j, i, 2]):
                        hex = RegularPolygon((xx[(j, i)]*2-0.5,
                                              yy[(j,i)]*2+(np.sqrt(3)/2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,2])),
                                             alpha=1)
                        ax.add_patch(hex)
                        

                    
                    if j==0:
                        # Central Hexagon
                        hex = RegularPolygon((xx[(j, i)]*2,
                                              yy[(j,i)]*2+(0.8660254*self.mapsize[1]*2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor= cm.jet(norm(umat[j][i])),
                                             alpha=1)#, edgecolor='black')
                        ax.add_patch(hex)
                        
                        # Right
                        if not np.isnan(um[j, i, 0]):
                            hex = RegularPolygon((xx[(j, i)]*2+1,
                                                  yy[(j,i)]*2+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,0])),
                                                 alpha=1)
                            ax.add_patch(hex)
                            
                        # Lower Right Hexagon
                        if not np.isnan(um[j, i, 1]):
                            hex = RegularPolygon((xx[(j, i)]*2+0.5,
                                                  yy[(j,i)]*2-(np.sqrt(3)/2)+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,1])),
                                                 alpha=1)
                            ax.add_patch(hex)
                            
                        # Bottom Left Hexagon
                        if not np.isnan(um[j, i, 2]):
                            hex = RegularPolygon((xx[(j, i)]*2-0.5,
                                                  yy[(j,i)]*2-(np.sqrt(3)/2)+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,2])),
                                                 alpha=1)
                            ax.add_patch(hex)
                    if i==0:
                        # Central Hexagon
                        hex = RegularPolygon((xx[(j, i)]*2 + self.mapsize[0]*2 - 1,
                                              yy[(j,i)]*2),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor= cm.jet(norm(umat[j][i])),
                                             alpha=1)#, edgecolor='red')
                        ax.add_patch(hex)
                        
                        # Right
                        if not np.isnan(um[j, i, 0]):
                            hex = RegularPolygon((xx[(j, i)]*2 + self.mapsize[0]*2,
                                                  yy[(j,i)]*2),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,0])),
                                                 alpha=1)
                            ax.add_patch(hex)
                                                 
                        # Upper Right Hexagon
                        if not np.isnan(um[j, i, 1]):
                            hex = RegularPolygon((xx[(j, i)]*2 + self.mapsize[0]*2 - 0.5,
                                                  yy[(j,i)]*2+(np.sqrt(3)/2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,1])),
                                                 alpha=1)
                            ax.add_patch(hex)
                            
                        # Bottom Left Hexagon
                        if not np.isnan(um[j, i, 2]):
                            hex = RegularPolygon((xx[(j, i)]*2 + self.mapsize[0]*2 - 1.5,
                                                  yy[(j,i)]*2+(np.sqrt(3)/2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,2])),
                                                 alpha=1)
                            ax.add_patch(hex)
                            
                    if i==0 and j==0:
                        # Central Hexagon
                        hex = RegularPolygon((xx[(j, i)]*2 + self.mapsize[0]*2-1,
                                              yy[(j,i)]*2+(0.8660254*self.mapsize[1]*2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor= cm.jet(norm(umat[j][i])),
                                             alpha=1)#, edgecolor='black')
                        ax.add_patch(hex)
                        
                        # Right
                        if not np.isnan(um[j, i, 0]):
                            hex = RegularPolygon((xx[(j, i)]*2 + self.mapsize[0]*2,
                                                  yy[(j,i)]*2+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,0])),
                                                 alpha=1)
                            ax.add_patch(hex)
                                                 
                       # Lower Right Hexagon
                        if not np.isnan(um[j, i, 1]):
                            hex = RegularPolygon((xx[(j, i)]*2+ self.mapsize[0]*2-0.5,
                                                  yy[(j,i)]*2-(np.sqrt(3)/2)+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,1])),
                                                 alpha=1)
                            ax.add_patch(hex)
                            
                        # Bottom Left Hexagon
                        if not np.isnan(um[j, i, 2]):
                            hex = RegularPolygon((xx[(j, i)]*2+ self.mapsize[0]*2-1.5,
                                                  yy[(j,i)]*2-(np.sqrt(3)/2)+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3),
                                                 facecolor=cm.jet(norm(um[j,i,2])),
                                                 alpha=1)
                            ax.add_patch(hex)

                    # Plot hits
                    if hits:
                        try:
                            hex = RegularPolygon((xx[(j,i)]*2,
                                                  yy[(j,i)]*2),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3)*bmu_dic[counter],
                                                 facecolor='white',
                                                 edgecolor='lightgray',
                                                 linewidth=1,
                                                 alpha=1)
                            ax.add_patch(hex)
                            
                            if j==0:
                                hex = RegularPolygon((xx[(j,i)]*2,
                                                  yy[(j,i)]*2+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3)*bmu_dic[counter],
                                                 facecolor='white',
                                                 edgecolor='lightgray',
                                                 linewidth=1,
                                                 alpha=1)
                                ax.add_patch(hex)
                                
                            if i==0:
                                hex = RegularPolygon((xx[(j,i)]*2 + self.mapsize[0]*2 - 1,
                                                  yy[(j,i)]*2),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3)*bmu_dic[counter],
                                                 facecolor='white',
                                                 edgecolor='lightgray',
                                                 linewidth=1,
                                                 alpha=1)
                                ax.add_patch(hex)
                            
                            if i==0 and j==0:
                                hex = RegularPolygon((xx[(j,i)]*2 + self.mapsize[0]*2 - 1,
                                                  yy[(j,i)]*2+(0.8660254*self.mapsize[1]*2)),
                                                 numVertices=6,
                                                 radius=1/np.sqrt(3)*bmu_dic[counter],
                                                 facecolor='white',
                                                 edgecolor='lightgray',
                                                 linewidth=1,
                                                 alpha=1)
                                ax.add_patch(hex)
                        except:
                            pass
                            
                    counter+=1

            # Plot Parameters
            
            plt.xlim(0.5, 2*self.mapsize[0]-0.5)
            plt.ylim(0, 2*self.mapsize[1]*0.8660254)

            f.tight_layout()
            ax.set_axis_off()
            plt.gca().invert_yaxis()
            plt.close()

            os.makedirs("Plots/U_matrix", exist_ok=True)


            filename = "Plots/U_matrix/plain_umat.jpg"
            f.savefig(filename, dpi=300, bbox_inches = "tight")

            # Read the saved image into a NumPy ndarray
            umat_plain = mpl.image.imread(filename)[30:-30,30:-30]
            
            
            return umat_plain
 
        else:
            # Plotting
            f = plt.figure(figsize=figsize, dpi=300)
            
            gs = gridspec.GridSpec(100, 100)
            ax = f.add_subplot(gs[:95, 0:90])
            ax.set_aspect('equal')

            # Normalize colors for all hexagons
            norm = mpl.colors.Normalize(vmin=np.nanmin(um), vmax=np.nanmax(um))
            counter = 0
            
            if watermark_neurons:
                # For use to plot number of BMUs
                nnodes = self.mapsize[0] * self.mapsize[1]
                grid_bmus = np.linspace(1,nnodes, nnodes).reshape(self.mapsize[1], self.mapsize[0])

            for j in range(self.mapsize[1]):
                for i in range(self.mapsize[0]):
                    # Central Hexagon
                    hex = RegularPolygon((xx[(j, i)]*2,
                                          yy[(j,i)]*2),
                                         numVertices=6,
                                         radius=1/np.sqrt(3),
                                         facecolor= cm.jet(norm(umat[j][i])),
                                         alpha=1)#, edgecolor='black')

                    ax.add_patch(hex)

                     # Right Hexagon
                    if not np.isnan(um[j, i, 0]):
                        hex = RegularPolygon((xx[(j, i)]*2+1,
                                              yy[(j,i)]*2),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,0])),
                                             alpha=1)
                        ax.add_patch(hex)

                    # Upper Right Hexagon
                    if not np.isnan(um[j, i, 1]):
                        hex = RegularPolygon((xx[(j, i)]*2+0.5,
                                              yy[(j,i)]*2+(np.sqrt(3)/2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,1])),
                                             alpha=1)
                        ax.add_patch(hex)

                    # Upper Left Hexagon
                    if not np.isnan(um[j, i, 2]):
                        hex = RegularPolygon((xx[(j, i)]*2-0.5,
                                              yy[(j,i)]*2+(np.sqrt(3)/2)),
                                             numVertices=6,
                                             radius=1/np.sqrt(3),
                                             facecolor=cm.jet(norm(um[j,i,2])),
                                             alpha=1)
                        ax.add_patch(hex)

                    if watermark_neurons:
                        # Central Hexagon
                        hex = RegularPolygon((xx[(j, i)]*2,
                                              yy[(j,i)]*2),
                                              numVertices=6,
                                              radius=2/np.sqrt(3),
                                              facecolor= "white",
                                              edgecolor='black',
                                              alpha=watermark_neurons_alfa)#, edgecolor='black')
                        ax.add_patch(hex)

                        ax.text(xx[(j,i)]*2, yy[(j,i)]*2, 
                                s=f"{int(grid_bmus[j,i])}", 
                                size = neurons_fontsize,
                                horizontalalignment='center', 
                                verticalalignment='center', 
                                color='black')

                    # Plot hits
                    if hits:
                        if label_plot:
                            ind_var = list(self.component_names).index(label_plot_name)
                            bool_val = self.data_denorm[:,ind_var]
                            bool_val = np.array([0 if number < 0.5 else 1 for number in bool_val])

                            if counter in self.bmus:
                                vars_in_hit =  np.array([ind for ind, value in enumerate(self.bmus) if value == counter])
                                bool_hit = bool_val[vars_in_hit]
                                bool_hit = statistics.mode(bool_hit)

                                facecolor_hits = "black" if bool_hit == 0 else 'white'
                            else:
                                pass
                        else:
                            facecolor_hits = 'white'

                        try:
                            hex = RegularPolygon((xx[(j, i)]*2,
                                                  yy[(j,i)]*2),
                                                 numVertices=6,
                                                 radius=((1/np.sqrt(3))*bmu_dic[counter]),
                                                 facecolor=facecolor_hits,
                                                 edgecolor=facecolor_hits,
                                                 linewidth=1.1,
                                                 alpha=1)
                            ax.add_patch(hex)
                        except:
                            pass
                    counter+=1

            if samples_label:
                if project_samples_label is not None:
                    samples_label_names = project_samples_label.index.tolist()
                    som_bmus = project_samples_label.BMU.values.tolist()
                    rep_samples_dic = select_keys(self.rep_sample(project=project_samples_label), som_bmus)
                else:
                    if samples_label_index == "all":
                        samples_label_names = self.sample_names
                        samples_label_index = np.arange(0, len(samples_label_names), 1)
                    else:
                        samples_label_index = np.array(samples_label_index)
                        samples_label_names = self.sample_names[samples_label_index]

                    som_bmus = self.bmus.astype(int)[samples_label_index]
                    rep_samples_dic = select_keys(self.rep_sample(), som_bmus+1)

                if save_labels_rep:
                    def filter_dictionary(dictionary, values):
                        filtered_dict = {}
                        for key, val in dictionary.items():
                            if isinstance(val, list):
                                matching_values = [item for item in val if item in values]
                                if matching_values:
                                    filtered_dict[key] = matching_values
                            else:
                                if val in values:
                                    filtered_dict[key] = val
                        return filtered_dict
                    
                    rep_samples_dic_filter = filter_dictionary(rep_samples_dic, samples_label_names)
                    with open('Results/Representative_samples_umatrix.txt', 'w', encoding='utf-8') as file:
                        for key, value in rep_samples_dic_filter.items():
                            if isinstance(value, list):
                                value = ', '.join(value)
                            file.write(f'BMU {key}: {value}\n')
                
                counter = 0
                for j in range(self.mapsize[1]):
                    for i in range(self.mapsize[0]):
                        try:
                            if isinstance(rep_samples_dic[counter+1], list):
                                total = len(rep_samples_dic[counter+1])
                                selected_samples = list(set(samples_label_names).intersection(set(rep_samples_dic[counter+1])))
                                sample_name, idx_sample = search_strings(selected_samples, rep_samples_dic[counter+1])
                                rep_sample_name = f"{sample_name}({idx_sample+1}/{total})"
                                                                
                            else:
                                rep_sample_name = rep_samples_dic[counter+1]
                            
                            hex = RegularPolygon((xx[(j, i)]*2,
                                                yy[(j,i)]*2),
                                                numVertices=6,
                                                radius=((1/np.sqrt(3))*0.5),
                                                facecolor='black',
                                                edgecolor='white',
                                                linewidth=1.1,
                                                alpha=1)
                            ax.add_patch(hex)

                            box_props = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round'}
                            ax.text(xx[(j,i)]*2, yy[(j,i)]*2+1.5*(1/np.sqrt(3)), 
                                    s=f"{rep_sample_name}", 
                                    size = samples_label_fontsize,
                                    horizontalalignment='center', 
                                    verticalalignment='top', 
                                    color='black',
                                    bbox= box_props)

                        except:
                            pass
                    
                        counter+=1
                    

            # Plot Parameters
            plt.xlim(-1.1, 2*self.mapsize[0]+0.1)
            plt.ylim(-0.5660254-0.6, 2*self.mapsize[1]*0.8660254-2*0.560254+0.6)
            ax.set_axis_off()
            plt.gca().invert_yaxis()

            # Map title
            plt.title(fill(title,30),
                      horizontalalignment='center',
                      verticalalignment='top',
                      size=title_size,
                      pad=title_pad)

            # Legend
            ax2 = f.add_subplot(gs[30:70, 95:98])
            cmap = mpl.cm.turbo
            norm = mpl.colors.Normalize(vmin=np.nanmin(um),
                                        vmax=np.nanmax(um))

            # Color bar
            cb1 = mpl.colorbar.ColorbarBase(ax2,
                                            cmap=cmap,
                                            norm=norm,
                                            orientation='vertical')
            # Legend Parameters
            cb1.ax.tick_params(labelsize=legend_ticks_size)

            
            cb1.set_label(fill(legend_title, 20), 
                          size=legend_title_size, 
                          labelpad=20)
            cb1.ax.yaxis.label.set_position(label_title_xy)
            """
            # Move the colorbar a little to the right
            pos = cb1.ax.get_position()
            pos.x0 += 0.08 * (pos.x1 - pos.x0)
            pos.x1 += 0.08 * (pos.x1 - pos.x0)
            cb1.ax.set_position(pos)
            
            cb1.ax.yaxis.label.set_position(label_title_xy)"""
            
            #ADD WATERMARK
            # Add white space subplot below the plot
            ax3 = f.add_subplot(gs[95:100, 0:20], zorder=-1)

            # Add the watermark image to the white space subplot
            ax3.imshow(self.foot, aspect='equal', alpha=1)
            ax3.axis('off')
            
            plt.subplots_adjust(bottom=3, top = 5,wspace=0)
            
            plt.show()
        
        if save:
            if file_path:
                f.savefig(f"{file_path}/{file_name}.jpg",dpi=300, bbox_inches = "tight")
            else:
                # Create directories if they don't exist
                path = 'Plots/U_matrix'
                os.makedirs(path, exist_ok=True)

                print("Saving.")
                if hits:
                    if label_plot:
                        f.savefig(f"Plots/U_matrix/{file_name}_with_hits_label.jpg",dpi=300, bbox_inches = "tight") 
                    elif watermark_neurons:
                        f.savefig(f"Plots/U_matrix/{file_name}_with_hits_watermarkneurons.jpg",dpi=300, bbox_inches = "tight") 
                    else:
                        f.savefig(f"Plots/U_matrix/{file_name}_with_hits.jpg",dpi=300, bbox_inches = "tight")
                else:
                    if label_plot:
                        f.savefig(f"Plots/U_matrix/{file_name}_with_label.jpg",dpi=300, bbox_inches = "tight") 
                    elif watermark_neurons:
                        f.savefig(f"Plots/U_matrix/{file_name}_watermarkneurons.jpg",dpi=300, bbox_inches = "tight") 
                    else:
                        f.savefig(f"Plots/U_matrix/{file_name}.jpg",dpi=300, bbox_inches = "tight")


    def component_plot(self,
                       component_name = 0,
                       figsize = (10,10),
                       title = None,
                       full_title = False,
                       title_size = 30,
                       title_pad = 25,
                       legend_title = False,
                       legend_pad = 0,
                       label_title_xy = (0,0.5),
                       legend_title_size = 24,
                       legend_ticks_size = 20,
                       save = False,
                       file_name = None,
                       file_path = False,
                       collage = False):
        """
        Function to perform the plotting of a trained variable map.

            Args:
                component_name: the name of the variable you want to plot. It is
                    accept the variable name in the form of a string or an integer
                    number related to the index of that variable in the initial
                    variables list.

                figsize: size of the variable's plot screen.
                    Default: (10,10)

                title: title of the created figure. Default: "U-Matrix"

                title_size: size of the plotted title. Default: 40

                title_pad: spacing between the title and the top of the map.
                    Default: 25

                legend_title: title of the color bar legend.
                    Default: "Distance"

                legend_pad: spacing between the legend title and the map.
                    Default: 0

                y_legend: spacing between legend title and bottom
                    of the figure. Default: 1.12

                legend_title_size = subtitle title size. Default: 24.

                legend_ticks_size = size of the color bar digits of the
                    subtitle. Default: 20.

                save: boolean to define the saving of the created image.
                    This saving will be done in the directory (Plots/Component_plots).
                    Default: True.

                file_name: the name that will be given to the saved file. If no name
                    is given, the name of the project will be used.

                file_path: the path on the system where the image should be
                    saved in case you do not choose to use the default path.

            Returns:
                The image plotting the variable map.
        """
        def check_path(filename):
            # Check if the file already exists in the directory
            if os.path.isfile(filename):
                # Extract the filename and extension
                name, ext = os.path.splitext(filename)

                # Append a number to the filename to make it unique
                i = 1
                while os.path.isfile(f"{name}_{i}{ext}"):
                    i += 1

                # Change the filename to the unique name
                filename = f"{name}_{i}{ext}"
            
            return filename
        
        if file_name is None:
            file_name = f"Component_plot_{self.name}"

        if isinstance(component_name, int):
            # Pegar uma variavel
            bmu_var = self.neuron_matrix[:,component_name].reshape(self.mapsize[1], self.mapsize[0])
            var_name = self.component_names[component_name]

            # Captar unidade da legenda
            if not legend_title:
                legend_title = self.unit_names[component_name]
                
        elif isinstance(component_name, str):
            index = list(self.component_names).index(component_name)
            bmu_var = self.neuron_matrix[:, index].reshape(self.mapsize[1], self.mapsize[0])
            var_name = self.component_names[index]
            # Captar unidade da legenda
            if not legend_title:
                legend_title = self.unit_names[index]
        else:
            print("Wrong name for component, accepts only string with component name or int with its position")

        # Create coordinates
        xx = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,0], (self.mapsize[1], self.mapsize[0]))
        yy = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,1], (self.mapsize[1], self.mapsize[0]))

        # Plotting
        f = plt.figure(figsize=figsize,dpi=300)
        gs = gridspec.GridSpec(100, 20)
        ax = f.add_subplot(gs[:95, 0:19])
        ax.set_aspect('equal')

        # Normalize colors for all hexagons
        norm = mpl.colors.Normalize(vmin=np.nanmin(bmu_var), vmax=np.nanmax(bmu_var))

        # Fill the plot with the hexagons
        for j in range(self.mapsize[1]):
            for i in range(self.mapsize[0]):
                ax.add_patch(RegularPolygon((xx[(j, i)], yy[(j,i)]),
                                     numVertices=6,
                                     radius=1/np.sqrt(3),
                                     facecolor=cm.jet(norm(bmu_var[j,i])),
                                     alpha=1))

        plt.xlim(-0.5, self.mapsize[0]+0.5)
        plt.ylim(-0.5660254, self.mapsize[1]*0.8660254+2*0.560254)
        ax.set_axis_off()
        plt.gca().invert_yaxis()

        # Map title
        if full_title:
            name = title if title is not None else var_name
        else:
            if title is not None:
                name = title
            else:
                name = var_name.split()
                name = f"{name[0]} {name[1]}" if len(name)>1 else f"{name[0]}"
        plt.title(fill(f"{name}",20), horizontalalignment='center',  verticalalignment='top', size=title_size, pad=title_pad)

        # Legend
        ax2 = f.add_subplot(gs[27:70, 19])
        cmap = mpl.cm.turbo
        norm = mpl.colors.Normalize(vmin=np.nanmin(bmu_var), vmax=np.nanmax(bmu_var))
        cb1 = mpl.colorbar.ColorbarBase(ax2,
                                        cmap=cmap,
                                        norm=norm,
                                        orientation='vertical')

        cb1.ax.tick_params(labelsize=legend_ticks_size)
        # Put the caption title if it has a variable name
        cb1.set_label(fill(legend_title,15),
                      size=legend_title_size,
                      labelpad=legend_pad,
                      horizontalalignment='right',
                      wrap=True)
        
        cb1.ax.yaxis.label.set_position(label_title_xy)
        if collage == False:
            #ADD WATERMARK
            # Add white space subplot below the plot
            ax3 = f.add_subplot(gs[95:100, 0:4], zorder=-1)

            # Add the watermark image to the white space subplot
            ax3.imshow(self.foot, aspect='equal', alpha=1)
            ax3.axis('off')
        
        plt.subplots_adjust(bottom=3, top = 5,wspace=0)
        if full_title:
            if isinstance(var_name, str):
                label_name = var_name.replace("/", "")
            else:
                label_name = var_name
        else:
            if isinstance(var_name, str):
                label_name = var_name[:7].replace(" ", "").replace(":", "")
            else:
                label_name = var_name[:6]

        if collage:
            path = 'Plots/Component_plots/Collage/temp'
            os.makedirs(path, exist_ok=True)
            path_name = check_path(f"Plots/Component_plots/Collage/temp/{label_name}.jpg")
            f.savefig(path_name,dpi=300, bbox_inches = "tight")

        if save:
            if file_path:
                path_name = check_path(f"{file_path}/{label_name}.jpg")
                f.savefig(path_name,dpi=300, bbox_inches = "tight")
            else:
                path = 'Plots/Component_plots'
                os.makedirs(path, exist_ok=True)
                path_name = check_path(f"Plots/Component_plots/{label_name}.jpg")
                f.savefig(path_name,dpi=300, bbox_inches = "tight")

    def multiple_component_plots(self,
                                wich = "all",
                                figsize = (10,10),
                                full_title = False,
                                title_size = 30,
                                title_pad = 25,
                                legend_title = "Presence",
                                legend_pad = 0,
                                label_title_xy = (0,0.5),
                                legend_title_size = 24,
                                legend_ticks_size = 20,
                                save = True,
                                file_path = False, 
                                collage = False):
        """
        Function for plotting a list or all variables trained in the
        SOM object.

        Args:
            wich: list of variables to be plotted and saved or "all" for
            plotting all variables.
        """
        if isinstance(wich, str):
            iterator = self.component_names
        else:
            iterator = wich

        # Iteração sobre a função de plotagem individual
        pbar = tqdm(iterator, mininterval=1)
        for name in pbar:
            pbar.set_description(f"Component: {name}")
            self.component_plot(component_name = name,
                               figsize = figsize,
                               full_title = full_title,
                               title_size = title_size,
                               title_pad = title_pad,
                               legend_title = legend_title,
                               legend_title_size = legend_title_size,
                               legend_ticks_size = legend_ticks_size,
                               legend_pad = legend_pad,
                               label_title_xy= label_title_xy,
                               save = save,
                               file_path = file_path,
                               collage = collage)
            plt.close()

        print("Finished")



    def component_plot_collage(self,
                               page_size = (2480, 3508), # A4 em pixels
                               grid = (4,4),
                               wich = "all",
                               figsize = (10,10),
                               full_title = False,
                               title_size = 30,
                               title_pad = 25,
                               legend_title = "Presence",
                               legend_title_size = 24,
                               legend_ticks_size = 20,
                               legend_pad = 0,
                               label_title_xy = (0,0.5),
                               file_path=False):
        """
        Function to create a collage of training component maps
        (Component Plots).

        Args:
            page_size: size of the plot collage page in pixels.
                Default: A4 (2480, 3508)

            grid: the format of the collage grid. Default: (4,4)

            wich: list of variables to be pasted and saved or "all" for
                plotting all variables.

            title_size: size of the plotted title. Default: 30

            title_pad: spacing between the title and the top of the array.
                Default: 25

            legend_title: title of the legend color bar.
                Default: "Presence"

            legend_pad: spacing between the legend title and the U matrix.
                Default: 0

            y_legend: spacing between legend title and bottom
                of the figure. Default: 1.12

            legend_title_size = subtitle title size. Default: 24.

            legend_ticks_size = size of the color bar digits of the
                subtitle. Default: 20.

            file_path = path to disired directory to save your collage.
        """
        # Support function only for using inside the function
        def resize_image(image, page, grid):
            """
           Function to resize the image on the sheet according to the defined
            grid.

            Args:
                image: the image that should be scaled in the grid.

                page: The page dimensions for the collage.

                grid: the format of the collage grid.
            """
            max_hor = page.size[0] / grid[1]
            max_ver = page.size[1] / grid[0]

            image.thumbnail((max_hor, max_ver))

            return image

        print("Generating maps...")
        # List component plots that haven't been made yet
        if isinstance(wich, str):
            if wich == "all":
                list_figs = "all"  
                n_components = len(self.component_names)
            else:
                print("The accepted parameters are 'all' or a list of variables or variable indexes.")
        if isinstance(wich, list):
            list_figs = wich
            n_components = len(wich)

        # Create component plots that have not been previously generated
        self.multiple_component_plots(wich = list_figs,
                                      figsize = figsize,
                                      full_title = full_title,
                                      title_size = title_size,
                                      title_pad = title_pad,
                                      save=False,
                                      legend_title = legend_title,
                                      legend_title_size = legend_title_size,
                                      legend_ticks_size = legend_ticks_size,
                                      legend_pad = legend_pad,
                                      label_title_xy = label_title_xy,
                                      collage = True)
        # Acquire the path for all images
        images_path = glob.glob('Plots/Component_plots/Collage/temp/*.jpg')

        # Number of pages needed to populate the components in the grid
        n_pages = int(n_components/(grid[0]*grid[1]))+1

        # Number of images per page
        im_pp = grid[0]*grid[1]

        print("Generating collage...")
        for i in range(n_pages):
            # Image to make the collage
            page = Image.new("RGB", page_size, "WHITE")  # White background

            # Slicing image paths for each page
            images_path_page = images_path[i*im_pp:(i+1)*im_pp]
            xx = np.tile(np.arange(grid[1]), grid[0])
            yy = np.repeat(np.arange(grid[0]), grid[1])

            for j, img_path in enumerate(images_path_page):
                img = Image.open(img_path) 
                img = resize_image(img, page, grid)
                x_pos = int(xx[j] * page_size[0] / grid[1])
                y_pos = int(yy[j] * page_size[1] / grid[0])
                page.paste(img, (x_pos, y_pos))

            # Save at plots root directory
            path = 'Plots/Component_plots/Collage/pages'
            os.makedirs(path, exist_ok=True)
            
            foot = self.foot
            max_height = page_size[1]/40
            width = int((foot.width / foot.height) * max_height)
            foot.thumbnail((width, max_height))
            x = 0  # Left position
            y = page.height - foot.height  # Down position
            page.paste(foot, (x, y))
            
            if file_path:
                page.save(f"{file_path}/Component_plots_collage_page{i+1}.jpg")
            else:
                page.save(f"Plots/Component_plots/Collage/pages/Component_plots_collage_page{i+1}.jpg")

        print("Finished.")
        print("The folder 'Plots/Component_plots/Collage/temp' can be deleted.")
    
    def bmu_template(self, 
                     figsize = (10,10),
                     title_size = 24,
                     fontsize = 10,
                     save = False,
                     file_name = None,
                     file_path = False):
        """
        Generates the BMU map for the current Kohonen map.        

        Args:
                figsize: size of the variable's plot screen.
                    Default: (10,10)

                title_size: size of the plotted title. Default: 24
                    
                fontsize: Default: 10

                save: boolean to define the saving of the created image.
                    This saving will be done in the directory (Plots/Bmu_template).
                    Default: False.

                file_name: the name that will be given to the saved file. If no name
                    is given, the name of the project will be used.

                file_path: the path on the system where the image should be
                    saved in case you do not choose to use the default path.
        """

        # Create coordinates
        xx = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,0], (self.mapsize[1], self.mapsize[0]))
        yy = np.reshape(self.generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,1], (self.mapsize[1], self.mapsize[0]))

        # Plotting
        f = plt.figure(figsize=figsize, dpi=300)
        gs = gridspec.GridSpec(100, 20)
        ax = f.add_subplot(gs[:, 0:20])
        ax.set_aspect('equal')

        # Normalize colors for all hexagons
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        
        # Create numbering
        nnodes = self.mapsize[0] * self.mapsize[1]
        grid_bmus = np.linspace(1,nnodes, nnodes).reshape(self.mapsize[1], self.mapsize[0])

        # Fill the plot with the hexagons
        for j in range(self.mapsize[1]):
            for i in range(self.mapsize[0]):
                hex = RegularPolygon((xx[(j,i)], yy[(j,i)]), 
                                     numVertices=6, 
                                     radius=1/np.sqrt(3), facecolor=[cm.Pastel1(norm(0.2)) if i%2==0 else cm.Pastel1(norm(0.6))][0],
                                     alpha= [0.3 if j%2==0 else 0.8][0], 
                                     edgecolor='gray'
                                    )
                ax.add_patch(hex)
                
                ax.text(xx[(j,i)], yy[(j,i)], 
                        s=f"{int(grid_bmus[j,i])}", 
                        size = fontsize,
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        color='black')
        
        plt.title("Neurons Template", size=title_size)
        
        # Limites de Plotagem e eixos
        plt.xlim(-0.5, self.mapsize[0]+0.5)
        plt.ylim(-0.5660254, self.mapsize[1]*0.8660254+2*0.560254)
        ax.set_axis_off()
        plt.gca().invert_yaxis()
        
        if save:
            if file_path:
                f.savefig(f"{file_path}/{file_name}_neurons_template.jpg",dpi=300, bbox_inches = "tight")
            else:
                path = 'Plots/Neurons_template'
                os.makedirs(path, exist_ok=True)

                f.savefig(f"Plots/Neurons_template/{file_name}_neurons_template.jpg",dpi=300, bbox_inches = "tight")

    def generate_rec_lattice(self, n_columns, n_rows):
        """
        Generates the xy coordinates of the BMUs for a rectangular grid.

        Args:
            n_rows: Number of lines in the kohonen map.
            n_columns: Number of columns in the kohonen map.

        Returns:
            Coordinates of the [x,y] format for the BMUs in a rectangular grid.

        """
        x_coord = []
        y_coord = []
        for j in range(n_rows):
            for i in range(n_columns):
                x_coord.append(i)
                y_coord.append(j)
        coordinates = np.column_stack([x_coord, y_coord])
        return coordinates

    @property
    def hits_dictionary(self):
        """
        Function to create a hits dictionary from the input vectors for
        each of its BMUs, proportional to the size of the plot.
        """
        # Hits count
        unique, counts = np.unique(self.bmus, return_counts=True)

        # Normalize this count from 0.5 to 2.0 (from a small hexagon to a
        # hexagon that covers half of the neighbors).
        counts = minmax_scale(counts, feature_range = (0.5,2))

        return dict(zip(unique, counts))


    def generate_hex_lattice(self, n_columns, n_rows):
        """
        Generates the xy coordinates of the BMUs for an odd-r hex grid.
        Args:
            n_rows: Number of lines in the kohonen map.
            n_columns: Number of columns in the kohonen map.

        Returns:
            Coordinates in the [x,y] format for the BMUs in a hexagonal grid.

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



    def plot_torus(self, inner_out_prop = 0.4, red_factor = 4, hits=False):
        """
        Returns the (x, y, z) corrdinates for drawing a toroid.
        
        Args:
        - rows (int): number of grid rows.
        - cols (int): number of grid columns.
        - aspect_ratio (float): proportion between the width and height of the image.
        - R_scale (float): external radius scale of toroid. Default: 0.4.

        Returns:
        (x, y, z) coordinates for drawing a toroid.
        """

        mat_im = self.plot_umatrix(figsize = (10,10), 
                                       hits = True if hits else False, 
                                       save = True, 
                                       file_name = None,
                                       file_path = False, 
                                       resume=True)
            
        # Toroid Resolution
        y_res = int(mat_im.shape[0]/red_factor)
        x_res = int(mat_im.shape[1]/red_factor)

        if mat_im.shape[0] > mat_im.shape[1]:
            mat_im = rotate(mat_im, 90)

        # Reduce image
        mat_res = (resize(mat_im, (y_res, x_res))*256).astype(int)

        def torus(rows, cols, aspect_ratio, R_scale=0.4):
            r_scale = R_scale * aspect_ratio * inner_out_prop
            u, v = np.meshgrid(np.linspace(0, 2*pi, cols), np.linspace(0, 2*pi, rows))
            return (R_scale+r_scale*np.sin(v))*np.cos(u), (R_scale+r_scale*np.sin(v))*np.sin(u), r_scale*np.cos(v)


        r, c, _ = mat_res.shape
        aspect_ratio = mat_im.shape[1]/mat_im.shape[0]
        x, y, z = torus(r, c, aspect_ratio, R_scale=0.4)
        I, J, K, tri_color_intensity, pl_colorscale = self.mesh_data(mat_res, n_colors=32, n_training_pixels=10000) 
        fig5 = go.Figure()
        fig5.add_mesh3d(x=x.flatten(), y=y.flatten(), z=np.flipud(z).flatten(),  
                                    i=I, j=J, k=K, intensity=tri_color_intensity, intensitymode="cell", 
                                    colorscale=pl_colorscale, showscale=False)
        
        scene_style = dict(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                   scene_aspectmode="data")
        
        fig5.update_layout(width=700, height=700,
                        margin=dict(t=10, r=10, b=10, l=10),
                        scene_camera_eye=dict(x=-1.75, y=-1.75, z=1), 
                        **scene_style)
        fig5.show()


    def mesh_data(self, img, n_colors=32, n_training_pixels=800):
        rows, cols, _ = img.shape
        z_data, pl_colorscale = self.image2zvals(img, n_colors=n_colors, n_training_pixels=n_training_pixels)
        triangles = self.regular_tri(rows, cols) 
        I, J, K = triangles.T
        zc = z_data.flatten()[triangles] 
        tri_color_intensity = [zc[k][2] if k%2 else zc[k][1] for k in range(len(zc))]  
        return I, J, K, tri_color_intensity, pl_colorscale
    
    def regular_tri(self, rows, cols):
        #define triangles for a np.meshgrid(np.linspace(a, b, cols), np.linspace(c,d, rows))
        triangles = []
        for i in range(rows-1):
            for j in range(cols-1):
                k = j+i*cols
                triangles.extend([[k,  k+cols, k+1+cols], [k, k+1+cols, k+1]])
        return np.array(triangles) 
    
    def image2zvals(self, img,  n_colors=64, n_training_pixels=800, rngs = 123): 
        # Image color quantization
        # img - np.ndarray of shape (m, n, 3) or (m, n, 4)
        # n_colors: int,  number of colors for color quantization
        # n_training_pixels: int, the number of image pixels to fit a KMeans instance to them
        # returns the array of z_values for the heatmap representation, and a plotly colorscale
    
        if img.ndim != 3:
            raise ValueError(f"Your image does not appear to  be a color image. It's shape is  {img.shape}")
        rows, cols, d = img.shape
        if d < 3:
            raise ValueError(f"A color image should have the shape (m, n, d), d=3 or 4. Your  d = {d}") 
            
        range0 = img[:, :, 0].max() - img[:, :, 0].min()
        if range0 > 1: #normalize the img values
            img = np.clip(img.astype(float)/255, 0, 1)
            
        observations = img[:, :, :3].reshape(rows*cols, 3)
        training_pixels = shuffle(observations, random_state=rngs)[:n_training_pixels]
        model = KMeans(n_clusters=n_colors, random_state=rngs).fit(training_pixels)
        
        codebook = model.cluster_centers_
        indices = model.predict(observations)
        z_vals = indices.astype(float) / (n_colors-1) #normalization (i.e. map indices to  [0,1])
        z_vals = z_vals.reshape(rows, cols)
        # define the Plotly colorscale with n_colors entries    
        scale = np.linspace(0, 1, n_colors)
        colors = (codebook*255).astype(np.uint8)
        pl_colorscale = [[sv, f'rgb{tuple(color)}'] for sv, color in zip(scale, colors)]
        
        # Reshape z_vals  to  img.shape[:2]
        return z_vals.reshape(rows, cols), pl_colorscale
