# External imports
import sys
import tempfile
import os
import itertools
import numpy as np
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
from scipy import sparse as sp
from joblib import load, dump
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
import pandas as pd
import json
from tqdm.auto import tqdm
from scipy.ndimage import shift

# Plots
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Rectangle
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib as mpl


# Internal imports
from .codebook import Codebook
from .object_functions import NeighborhoodFactory, NormalizerFactory


class SOMFactory(object):
    @staticmethod
    def build(data,
              mapsize=None,
              mask=None,
              mapshape='toroid',
              lattice='hexa',
              normalization='var',
              initialization='random',
              neighborhood='gaussian',
              training='batch',
              name='IntraSOM',
              component_names=None,
              unit_names = None,
              sample_names=None,
              missing=False,
              save_nan_hist=False,
              pred_size=0):
        """

         onstructs an object for SOM training, with the data parameters,
        map features and types of training.

        Args:
            data: input data, represented by an n-row matrix, as samples or instances, 
                and m columns, as variables. The accepted formats are dataframe or ndarray 
                (if choosing the ndarray format, the parameters `component_names` and 
                `sample_names` can be filled).

            mapsize: tuple/list defining the size of the SOM map in the format (columns, 
                rows). If an integer is provided, it is considered as the number of neurons. 
                For example, for a map of 144 neurons, a SOM map (12, 12) will be 
                automatically created. For the development of periodicity in hexagonal 
                grids, it is not possible to create SOM maps with an odd number of rows. 
                Therefore, when choosing map sizes with an odd number of rows, the 
                immediately lower even integer will be automatically admitted. If no 
                number is entered, the size provided by the heuristic function defined 
                in expected_mapsize() will be considered.

            mask: Mask for null values. Example: -9999.

            mapshape: Format of the SOM topology. Example: "planar" or "toroid".

            lattice: type of lattice. Example: "hexa".

            normalization: type of data normalization. Example: "var" or None

            initialization: method used for SOM initialization. Options: "pca" (only 
                for complete datasets without NaN values; does not work with missing 
                data) or "random".

            neighborhood: type of neighborhood calculation. Example: "gaussian"

            training: type of neighborhood calculation. Example: "gaussian"

            name: name used to identify the SOM object or project. The chosen 
                name will be used to name the saved files at the end of training 
                and in other functions of the library.

            component_names: list of labels for the variables used in training. 
                If not provided, a list will be automatically created in the 
                format: [Variable 1, Variable 2, ...].

            unit_names: list of labels associated with the units of the training 
                variables. If not provided, a unit list will be automatically created 
                in the style: [Unit , Unit , ...].

            sample_names: List with the names of the samples. If not provided, a 
                list will be automatically created in the format: [Sample 1, 
                Sample 2, ...].

            missing: boolean value that should be filled if the database has missing 
                values (NaN). For training of the "Bruto" type, a search for the BMUs 
                (Best Matching Units) is performed using a distance calculation function 
                with missing data, and the codebook update is done by filling the missing 
                data with 0. In the fine-tuning step, this process is repeated if the 
                parameter "previous_epoch" is set to False, or there is a substitution 
                of the empty values with the calculated values for those cells 
                in the previous training epoch if the parameter "previous_epoch" is 
                set to True. In order to allow freedom of movement for vectors with 
                missing data across the Kohonen map, a random regularization factor 
                is generated for this filling. This factor decays during training 
                based on the decay of the search radius for BMU updates. This factor 
                can be observed along with the quantization error and the search 
                radius in the training progress bar.

            pred_size: for semi-supervised training of databases, it is recommended 
                to place the training label columns in the last positions of the 
                DataFrame. Please indicate here the number of labeled columns so 
                that the project_nan_data() projection function can be used with 
                unlabeled data.

        Returns:
            SOM object with all its inherited methods and attributes.

        """
        # Apply normalization if it is defined
        if normalization:
            normalizer = NormalizerFactory.build(normalization)
        else:
            normalizer = None

        # Build the neighborhood calculation object according to the function of
        # specified neighborhood
        neigh_calc = NeighborhoodFactory.build(neighborhood)

        return SOM(data = data,
                   neighborhood = neigh_calc,
                   normalizer = normalizer,
                   normalization = normalization,
                   mapsize = mapsize,
                   mask = mask,
                   mapshape =mapshape,
                   lattice = lattice,
                   initialization = initialization,
                   training = training,
                   name = name,
                   component_names = component_names,
                   unit_names = unit_names,
                   sample_names = sample_names,
                   missing = missing,
                   save_nan_hist=save_nan_hist,
                   pred_size = pred_size)

    @staticmethod
    def load_som(data,
             trained_neurons,
             params):
        """
        Function for loading trained data. It requires access to the input dataframes,
        the trained neuron dataframe, as well as the parameter file saved at the 
        end of the training process, as demonstrated in the example notebook.
        
        Args:
            data: Input data, represented by an n-row and m-column matrix as 
                variables. Accepts dataframe or ndarray format. If ndarray is 
                used, the component_names and sample_names variables can be filled.

            trained_neurons: The dataframe of trained neurons generated at the end 
                of SOM training.

            params: JSON parameters generated at the end of SOM training.

        Returns:
            Trained SOM object with all its inherited methods and attributes.
        """
        print("Loading data...")
        normalization = params["normalization"]
        if normalization:
            normalizer = NormalizerFactory.build(normalization)
        else:
            normalizer = None
        neigh_calc = NeighborhoodFactory.build(params["neighborhood"])
        mapsize = params["mapsize"]
        mask = params["mask"]
        mapshape = params["mapshape"]
        lattice = params["lattice"]
        initialization = params["initialization"]
        training = params["training"]
        name = params["name"]
        component_names = params["component_names"]
        unit_names = params["unit_names"]
        sample_names = params["sample_names"]
        missing = params["missing"]
        missing_imput = params["missing_imput"] if params["missing"]==True else None
        pred_size = params["pred_size"]
        bmus_ind = params["bmus"]
        bmus_dist = params["bmus_dist"]
        bmus = np.array([bmus_ind, bmus_dist])
        load_param=True

        return SOM(data = data,
                   neighborhood = neigh_calc,
                   normalizer = normalizer,
                   normalization = normalization,
                   mapsize = mapsize,
                   mask = mask,
                   mapshape = mapshape,
                   lattice = lattice,
                   initialization = initialization,
                   training = training,
                   name = name,
                   component_names = component_names,
                   unit_names = unit_names,
                   sample_names = sample_names,
                   missing = missing,
                   missing_imput = missing_imput,
                   pred_size = pred_size,
                   load_param = load_param,
                   trained_neurons = trained_neurons, 
                   bmus = bmus)

class SOM(object):

    def __init__(self,
                 data,
                 neighborhood='gaussian',
                 normalizer="var",
                 normalization = "var",
                 mapsize=None,
                 mask=None,
                 mapshape='toroid',
                 lattice='hexa',
                 initialization='random',
                 training='batch',
                 name='IntraSOM',
                 component_names=None,
                 unit_names = None,
                 sample_names=None,
                 missing=False,
                 save_nan_hist=False,
                 missing_imput=None,
                 pred_size=0,
                 load_param=False,
                 trained_neurons=None, 
                 bmus = None):

        # Mask for missing values
        self.mask = mask

        # Check input type and fill in internal attributes
        print("Loading dataframe...")
        if isinstance(data, pd.DataFrame):
            self.data_raw = data.values
            if missing:
                self.data_raw = np.where(self.data_raw == self.mask, np.nan, self.data_raw)
            self._data = normalizer.normalize(self.data_raw) if normalizer else self.data_raw
            self._component_names = np.array(component_names) if component_names else np.array(data.columns)
            self._sample_names = np.array(sample_names) if sample_names else np.array(data.index)
        
        elif isinstance(data, np.ndarray):
            self.data_raw = data
            if missing:
                self.data_raw[self.data_raw==self.mask]=np.nan
            self._data = normalizer.normalize(self.data_raw) if normalizer else self.data_raw
            self._component_names = np.array(component_names) if component_names else np.array([f"Var_{i}" for i in range(1,data.shape[1]+1)])
            if data.shape[0] < 10000:
                self._sample_names = np.array(sample_names) if sample_names else np.array([f"Sample_{i}" for i in range(1,data.shape[0]+1)])
            else:
                self._sample_names = np.array(sample_names) if sample_names else np.array([f"{i}" for i in range(1,data.shape[0]+1)])
        else:
            print("Only DataFrame or ndarray types are accepted as\
             IntraSOM inputs")

        # Populate non-type dependent attributes
        print("Normalizing data...")
        self._normalizer = normalizer
        self._normalization = normalization
        self._dim = data.shape[1]
        self._dlen = data.shape[0]
        self.pred_size = pred_size
        self.name = name
        self.missing = missing
        if self.missing == False:
            if np.isnan(self._data).any():
                sys.exit("Database with missing data, flag in missing parameter")
        print("Creating neighborhood...")
        self.neighborhood = neighborhood
        self._unit_names = unit_names if unit_names else [f"Unit {var}" for var in self._component_names]
        self.mapshape = mapshape
        self.initialization = initialization
        
        if mapsize:
            if mapsize[1]%2!=0:
                self.mapsize = (mapsize[0], mapsize[1]+1)
                print(f"The number of lines cannot be odd.\
                The map size has been changed to: {self.mapsize}")
            else:
                self.mapsize = mapsize
        else:
            self.mapsize = self._expected_mapsize(self._data)

        self.lattice = lattice
        self.training = training
        self.load_param = load_param
        self.save_nan_hist = save_nan_hist
        if save_nan_hist:
            self.nan_value_hist = []
        self.data_proj_norm = []

        # Populate load type dependent attributes
        if load_param:
            print("Creating missing data database")
            self.data_missing = {"indices":tuple(zip(*np.argwhere(np.isnan(self._data)))), 
                                 "nan_values":missing_imput}
            # Modificar nome
            self.name = self.name+"_loaded"
            # Alocando os bmus
            self._bmu = bmus
            print("Creating codebook...")
            self.codebook = Codebook(self.mapsize, self.lattice, self.mapshape)
            self.codebook.matrix = self._normalizer.normalize_by(self.data_raw, trained_neurons.iloc[:,7:].values)
            
            try:
                print("Loading distances matrix...")
                self._distance_matrix = np.load("Results/distance_matrix.npy")
                if self.mapsize[0]*self.mapsize[1] != self._distance_matrix.shape[0]:
                    self._distance_matrix = self.calculate_map_dist
            except:
                self._distance_matrix = self.calculate_map_dist
        else:
            self.data_missing = {"indices":tuple(zip(*np.argwhere(np.isnan(self.data_raw)))), 
                                 "nan_values":None}
            self._bmu = np.zeros((2,self._dlen))
            self.codebook = Codebook(self.mapsize, self.lattice, self.mapshape)
            try:
                self._distance_matrix = np.load("Results/distance_matrix.npy")
                if self.mapsize[0]*self.mapsize[1] != self._distance_matrix.shape[0]:
                    self._distance_matrix = self.calculate_map_dist
            except:
                self._distance_matrix = self.calculate_map_dist

    # CLASS PROPERTIES
    
    @property
    def get_data(self):
        """
        Class property function to return a copy of the input data with the missing data.
        """
        if self.missing:
            self._data[self.data_missing["indices"]] = np.nan
        else:
            pass
        return self._data
    
    @property
    def params_json(self):
        """
        Class property function to generate a csv file with the
        training parameters.
        """
        def fix_serialize(obj):
            if isinstance(obj, dict):
                for key in obj:
                    obj[key] = fix_serialize(obj[key])
                return obj
            elif isinstance(obj, list):
                return [fix_serialize(item) for item in obj]
            elif isinstance(obj, np.int64):
                return int(obj)
            else:
                return obj
            
        # Create the training properties dictionary
        dic = dict()
        dic["mapsize"] = self.mapsize
        if self.mask is not None:
            dic["mask"] = int(self.mask)
        else:
            dic["mask"] = self.mask
        dic["mapshape"] = self.mapshape
        dic["lattice"] = self.lattice
        dic["neighborhood"] = self.neighborhood.name
        dic["normalization"] = self._normalization
        dic["initialization"] = self.initialization
        dic["training"] = self.training
        dic["name"] = self.name
        dic["component_names"] = list(self._component_names)
        dic["unit_names"] = list(self._unit_names)
        dic["sample_names"] = list(self._sample_names)
        dic["missing"] = self.missing
        dic["pred_size"] = int(self.pred_size)
        dic["bmus"] = self._bmu[0].astype(int).tolist()
        if self.missing == True:
            dic["bmus_dist"] = self._bmu[1].tolist()
            dic["missing_imput"] = list(self.data_missing["nan_values"])
        elif self.missing == False:
            fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(self.get_data, nan=0.0), np.nan_to_num(self.get_data, nan=0.0))
            dic["bmus_dist"] = np.sqrt(self._bmu[1] + fixed_euclidean_x2).tolist()
            
        
        # Fix serialization problems
        dic = fix_serialize(dic)

        # Transform to JSON
        json_params = json.dumps(dic)

        # Save the result into the specified directory
        f = open(f"Results/params_{self.name}.json","w")
        f.write(json_params)
        f.close()

    @property
    def component_names(self):
        """
        Return variable names.
        """
        return self._component_names

    @property  
    def calculate_map_dist(self):
        """
        Calculates the grid distances, which will be used during the training
        steps and returns them in the form of an array of internal distances
        from the grid.
        """
        blen = 50
        
        # Capture the number of training neurons
        nnodes = self.codebook.nnodes

        # Create a matrix of zeros in the format nnodes x nnodes
        distance_matrix = np.zeros((nnodes, nnodes))
        
        # Iterates over the nodes and fills the distance matrix for each node,
        # through the grid_dist function
        print("Initializing map...")
        
        # Access matrix for the first neuron
        initial_matrix = self.codebook.grid_dist(0).reshape(self.mapsize[1], self.mapsize[0])
        
        counter = 0
        for i in tqdm(range(self.mapsize[1]), position=0, leave=True, desc="Creating Neuron Distance Rows", unit="rows"):
            for j in range(self.mapsize[0]):
                shifted = shift(initial_matrix, (i,j), mode='grid-wrap')
                #account for odd-r shifts - j direction
                if i%2!=0:
                    shifted[0::2] = shift(shifted[0::2], (0,1), mode='grid-wrap')
                distance_matrix[counter] = shifted.flatten().astype(int)
                counter+=1
        """
        for i in tqdm(range(nnodes), desc="Matriz\
            de distâncias", unit=" Neurons"):
            dist = self.codebook.grid_dist(i)
            distance_matrix[i:,i] = dist[i:]
            distance_matrix[i,i:] = dist[i:]
            del dist
        # Create directories if they don't exist
        path = 'Results'
        os.makedirs(path, exist_ok=True)
        # Save so that this process is done only 1x
        np.save('Results/distance_matrix.npy', distance_matrix) 
        """
        
        return distance_matrix

    @property
    def neuron_matrix(self):
        """
        Retorna a matriz de neurônios denormalizada. No formato de array com os
        valores dos vetores de cada neurônio.
        Returns the denormalized matrix of neuronss. In array format with the
        vector values ​​of each neuron.
        """
        # Differentiate the way of loading if it is a loading of
        # trained data  
        norm_neurons = self._normalizer.denormalize_by(self.data_raw, self.codebook.matrix)

        # Set a threshold for values ​​near zero
        threshold = 1e-6

        # Transform values near zero to zero
        transformed_neurons = np.where(np.abs(norm_neurons) < threshold, 0, norm_neurons)
        
        return transformed_neurons

    @property
    def neurons_dataframe(self):
        """
        Function to create a dataframe of the weights of neurons resulting from training. Returned
        in the form of a DataFrame of the BMU and their rectangular
        and cubic coordinates.
        """

        # Create dataframe
        neuron_df = pd.DataFrame(np.round(self.neuron_matrix,6),
                        index = list(range(1, self.neuron_matrix.shape[0]+1)),
                        columns=[f"B_{var}" for var in self._component_names])

        # Capture the number of columns and lines of the created map
        rows = self.mapsize[1]
        cols = self.mapsize[0]

        # Create rectangular and cubic coordinates
        rec_coordinates = self._generate_rec_lattice(cols, rows)
        cub_coordinates = self._generate_oddr_cube_lattice(cols, rows)
        
        # Scaling
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

        # Create columns
        neuron_df.insert(0, "Cub_z", cub_coordinates[:,2])
        neuron_df.insert(0, "Cub_y", cub_coordinates[:,1])
        neuron_df.insert(0, "Cub_x", cub_coordinates[:,0])
        neuron_df.insert(0, "Ret_y", rec_coordinates[:,1])
        neuron_df.insert(0, "Ret_x", rec_coordinates[:,0])
        neuron_df.insert(0, "Udist", min_max_scaler.fit_transform(self.build_umatrix().reshape(-1, 1)))    
        neuron_df.insert(0, "BMU", list(range(1, self.neuron_matrix.shape[0]+1)))

        return neuron_df.astype({"BMU": int,
                              "Ret_x": int,
                              "Ret_y": int,
                              "Cub_x": int,
                              "Cub_y": int,
                              "Cub_z": int,
                              "Udist": np.float32
                                  })

    @property
    def results_dataframe(self):
        """
        Function to create a dataframe with the BMU and the associated values to
        each input vector.
        """
        # Rescue the neuron dataframe
        bmu_df = self.neurons_dataframe
        bmus = self._bmu[0].astype(int)

        results_df = bmu_df.iloc[bmus,:]

        # Enter the quantization error for each vector
        results_df.insert(1, "q-error", self.calculate_quantization_error_expanded)

        # Change index with the sample names
        results_df.set_index(self._sample_names, inplace=True)

        # Regularize the data type
        return results_df.astype({"BMU": int,
                                   "Ret_x": int,
                                   "Ret_y": int,
                                   "Cub_x": int,
                                   "Cub_y": int,
                                   "Cub_z": int,
                                  "q-error": np.float32
                                  })

    @property
    def training_summary(self):
        """
        Function to create a training summary and save it in .txt format.
        """

        # Dictionary to make the terms more explanatory
        dic_params={
            "var":"Variance",
            "toroid":"Toroid",
            "hexa":"Hexagonal",
            "random":"Randomic",
            "gaussian":"Gaussian",
            True:"Yes"
        }

        # Open a text file
        text_file = open(f"Intrasom_report_{self.name}.txt", mode="w", encoding='utf-8')

        # Write the lines of text
        # Project variables
        text_file.write(f'IntraSOM Training Report\n')
        text_file.write(f'Project: {self.name}\n')
        text_file.write(f"\n")
        text_file.write(f"Input Data:\n")
        text_file.write(f"Features: {self._component_names.shape[0]}\n")
        text_file.write(f"Samples: {self._sample_names.shape[0]}\n")
        text_file.write(f"Cells: {self._sample_names.shape[0]*self._component_names.shape[0]}\n")
        if self.missing:
            text_file.write(f"Missing Cells: {np.isnan(self.data_raw).sum()}\n")
        text_file.write(f"\n")
        text_file.write(50*"-")
        text_file.write(f"\n")

        # Initialization Parameters
        text_file.write(f"Initialization Parameters:\n")
        text_file.write(f"\n")
        text_file.write(f"Map Size: {self.mapsize[0]} columns and {self.mapsize[1]} lines\n")
        if self.mask:
            text_file.write(f"Missing Mask: {self.mask}\n")
        text_file.write(f"Training Polygon: {dic_params.get(self.mapshape)}\n")
        text_file.write(f"Lattice: {dic_params.get(self.lattice)}\n")
        text_file.write(f"Normalization: {dic_params.get(self._normalizer.name)}\n")
        text_file.write(f"Initialization: {dic_params.get(self.initialization)}\n")
        text_file.write(f"Neighborhood Function: {dic_params.get(self.neighborhood.name)}\n")
        if self.missing:
            text_file.write(f"Missing Data: {dic_params.get(self.missing)}\n")
            text_file.write(f"Missing Data Percentage: \
            {round(np.isnan(self.data_raw).sum()/self.data_raw.flatten().shape[0]*100, 2)}%\n")

        text_file.write(f"Number of Labels: {self.pred_size}\n")
        text_file.write(f"\n")
        text_file.write(50*"-")
        text_file.write(f"\n")

        # Training Parameters
        text_file.write(f"Training Parameters:\n")
        text_file.write(f"\n")
        text_file.write(f"Rough Training:\n")
        text_file.write(f"Size: {self.train_rough_len}\n")
        text_file.write(f"Initial Ratio: {self.train_rough_radiusin}\n")
        text_file.write(f"Final Ratio: {self.train_rough_radiusfin}\n")
        text_file.write(f"\n")
        text_file.write(f"Finetube Training:\n")
        text_file.write(f"Size: {self.train_finetune_len}\n")
        text_file.write(f"Initial Ratio: {self.train_finetune_radiusin}\n")
        text_file.write(f"Final Ratio: {self.train_finetune_radiusfin}\n")
        text_file.write(f"\n")
        text_file.write(50*"-")
        text_file.write(f"\n")

        # Training Quality Parameters
        text_file.write(f"Training Evaluation:\n")
        text_file.write(f"\n")
        text_file.write(f"Quantization Error: {round(self.calculate_quantization_error, 4)}\n")
        text_file.write(f"Topographic Error: {round(self.topographic_error, 4)}\n")
        text_file.close()
        print("Training Report Created")


    # CLASS METHODS
    def imput_missing(self, save=True, round_values=False):
        """
        Returns data with imputed values ​​in empty input cells.

        Args:
            save: boolean value to indicate if the created file will be saved or not
            saved inside the [Imputation] directory.

        Returns:
            DataFrame with input data with imputed empty cells
                by their respective BMU.
        """
        def minimum_decimal_places(array):
            """
            Function to find the number of decimal places in each column of an array.
            """
            min_decimal_places = np.inf * np.ones(array.shape[1], dtype=int)

            # Iterate over each column
            for column in range(array.shape[1]):
                # Iterate over each number in the column
                for number in array[:, column]:
                    # Convert the number to a string
                    number_str = str(number)

                    # Check if the number is a decimal
                    if '.' in number_str:
                        # Get the decimal places count
                        decimal_places = len(number_str.split('.')[-1])
                        # Update the minimum decimal places for the column if necessary
                        min_decimal_places[column] = min(min_decimal_places[column], decimal_places)

            return min_decimal_places.astype(int)-1
        
        # Capture the data
        data = self.get_data
        data_folder = tempfile.mkdtemp()
        data_name = os.path.join(data_folder, 'data')
        dump(data, data_name)
        data = load(data_name, mmap_mode='r+')

        # Fill in
        data[self.data_missing["indices"]] = self.data_missing["nan_values"]

        # Denormalize
        data = self._normalizer.denormalize_by(self.data_raw, data)
        if round_values:
            # Round to the minimum of decimal places of each training column
            min_dec = minimum_decimal_places(self.data_raw)

            # Iterate over each column and round the values ​​with the corresponding decimal
            for i in range(data.shape[1]):
                data[:, i] = np.round(data[:, i], min_dec[i])
        else:
            data = np.round(data, decimals=6)

        # Replace the -0 with 0
        data = np.where((data == -0) | (data == -0.0), 0, data)

        # Create dataframe
        imput_df = pd.DataFrame(data, columns = self._component_names, index = self._sample_names)
        if save:
            # Create directories if they don't exist
            path = 'Imputation'
            os.makedirs(path, exist_ok=True)
            
            # Save
            imput_df.to_excel(f"Imputation/Imputed_data_{self.name}.xlsx")
            imput_df.to_csv(f"Imputation/Imputed_data_{self.name}.csv")

        return imput_df

    def project_nan_data(self,
                        data_proj,
                        with_labels=False,
                        sample_names=None,
                        save = True):
        """
        Function to project new data into the trained model, even if these
        data has missing values.

        Args:
            data_proj: Data that you want to project into the model. It may be in
                DataFrame or numpy ndarray format.

            with_labels: Boolean value to indicate if the data has the columns
                of labels (semi-supervised classification model) or not.

        Returns:
            DataFrame with the BMU representing each input vector.
        """
        
        # Check formats for adaptation
        if isinstance(data_proj, pd.DataFrame):
            sample_names = sample_names if sample_names is not None else data_proj.index.values
            data_proj = data_proj.values
        elif isinstance(data_proj, np.ndarray):
            data_proj = data_proj
            sample_names = sample_names if sample_names is not None else [f"Sample_proj_{i}" for i in range(1,data_proj.shape[0]+1)]
        else:
            print("Only DataFrame and ndarray formats are accepted as input")

        # Check for the presence of training labels in the data to be projected
        if with_labels:
            # Remove the label variables from the data
            data_proj = data_proj[:, :- self.pred_size]
            data_proj = self._normalizer.normalize_by(self.data_raw,
                                                      data_proj,
                                                      with_labels=True,
                                                      pred_size=self.pred_size)
        else:
            data_proj = data_proj
            data_proj = self._normalizer.normalize_by(self.data_raw,
                                                      data_proj,
                                                      with_labels=False,
                                                      pred_size=self.pred_size)

        self.data_proj_norm = data_proj

        # Find the BMU for this new data
        bmus = self._find_bmu(data_proj, project=True)
        bmus_ind = bmus[0].astype(int)

        # Rescue the BMU dataframe
        bmu_df = self.neurons_dataframe

        # Create dataframe for projection
        projected_df = bmu_df.iloc[bmus_ind,:]

        projected_df.set_index(np.array(sample_names), inplace=True)

        projected_df = projected_df.astype({"BMU": int,
                                            "Ret_x": int,
                                            "Ret_y": int,
                                            "Cub_x": int,
                                            "Cub_y": int,
                                            "Cub_z": int
                                           })
        # Save
        if save:
            # Create results folder
            try:
                os.mkdir("Projected")
            except:
                pass

            # Save
            projected_df.to_excel(f"Projected/Projected_data_{self.name}.xlsx")
            projected_df.to_csv(f"Projected/Projected_data_{self.name}.csv")

        return projected_df

    def denorm_data(self, data):
        """
        Class method to denormalize data according to normalization
        made for the input data.
        """
        data_denorm = self._normalizer.denormalize_by(self.data_raw, data)

        return data_denorm


    def train(self,
              bootstrap = False,
              bootstrap_proportion = 0.8,
              n_job=-1,
              save=True,
              summary=True,
              dtypes = "parquet",
              shared_memory=False,
              train_rough_len=None,
              train_rough_radiusin=None,
              train_rough_radiusfin=None,
              train_finetune_len=None,
              train_finetune_radiusin=None,
              train_finetune_radiusfin=None,
              train_len_factor=1,
              maxtrainlen=1000, 
              history_plot = False,
              previous_epoch = False):
        """
        Class method for training the SOM object.

        Args:
            n_job: number of jobs to use and parallelize training.

            shared_memory: flag to enable shared memory.

            train_rough_len: number of iterations during rough training.

            train_rough_radiusin: initial BMU fetching radius during
                rough training.

            train_rough_radiusfin: BMU search final radius during
                rough training.

            train_finetune_len: number of iterations during fine training.

            train_finetune_radiusin: initial BMU scan radius during
                fine training.

            train_finetune_radiusfin: BMU search final radius during
                fine training.

            train_len_factor: factor that multiplies the values ​​of the training
                extension (rough, fine, etc)

            maxtrainlen: maximum value of desired interactions.
                Default: np.Inf (infinity).

        Returns:
            SOM object trained according to the chosen parameters.

        """
        # Create training-related class attributes
        self.train_rough_len = train_rough_len
        self.train_rough_radiusin = train_rough_radiusin
        self.train_rough_radiusfin = train_rough_radiusfin
        self.train_finetune_len = train_finetune_len
        self.train_finetune_radiusin = train_finetune_radiusin
        self.train_finetune_radiusfin = train_finetune_radiusfin
        self.summary = summary
        self.save = save
        self.total_radius = train_rough_radiusin
        self.history_plot = history_plot
        self.actual_train = None
        self.bootstrap = bootstrap
        self.bootstrap_proportion = bootstrap_proportion
        self.previous_epoch = previous_epoch

        print("Starting Training...")

        # Apply the chosen startup type
        if self.missing:
            if self.load_param:
                self.codebook.pretrain()
            elif self.initialization == 'random':
                self.codebook.random_initialization(self.get_data)
            elif self.initialization == 'pca':
                print("Not implemented yet")
        else:
            if self.load_param:
                self.codebook.pretrain()
            elif self.initialization == 'random':
                self.codebook.random_initialization(self.get_data)
            elif self.initialization == 'pca':
                self.codebook.pca_linear_initialization(self.get_data)

        # Apply the chosen training type
        if self.training == 'batch':
            print("Rough Training:")
            self.actual_train = "Rough"
            self.rough_train(njob=n_job,
                             shared_memory=shared_memory,
                             trainlen=train_rough_len,
                             radiusin=train_rough_radiusin,
                             radiusfin=train_rough_radiusfin,
                             train_len_factor=train_len_factor,
                             maxtrainlen=maxtrainlen)
            
            print("Fine Tuning:")
            self.actual_train = "Fine"
            self.finetune_train(njob=n_job,
                                shared_memory=shared_memory,
                                trainlen=train_finetune_len,
                                radiusin=train_finetune_radiusin,
                                radiusfin=train_finetune_radiusfin,
                                train_len_factor=train_len_factor,
                                maxtrainlen=maxtrainlen)
            
            
            if self.save:
                print("Saving...")

                # Create directories if they don't exist
                path = 'Results'
                os.makedirs(path, exist_ok=True)

                # Save the results
                if dtypes == "xlsx_csv":
                    self.results_dataframe.to_excel(f"Results/{self.name}_results.xlsx")
                    self.neurons_dataframe.to_excel(f"Results/{self.name}_neurons.xlsx")
                    self.results_dataframe.to_csv(f"Results/{self.name}_results.csv")
                    self.neurons_dataframe.to_csv(f"Results/{self.name}_neurons.csv")
                elif dtypes == "xlsx":
                    self.results_dataframe.to_excel(f"Results/{self.name}_results.xlsx")
                    self.neurons_dataframe.to_excel(f"Results/{self.name}_neurons.xlsx")
                elif dtypes == "csv":
                    self.results_dataframe.to_csv(f"Results/{self.name}_results.csv")
                    self.neurons_dataframe.to_csv(f"Results/{self.name}_neurons.csv")
                elif dtypes == "parquet":
                    self.results_dataframe.to_parquet(f"Results/{self.name}_results.parquet")
                    self.neurons_dataframe.to_parquet(f"Results/{self.name}_neurons.parquet")
                else:
                    print("Chosen save type is incorrect.")
            if self.summary:
                self.training_summary

            # Create json for training parameters
            self.params_json

            print("Training completed successfully.")

        elif self.training == 'seq':
            print("Not implemented yet")
        else:
            print("The chosen training type is not in the acceptable list: 'batch' or 'seq'")

    def _calculate_ms_and_mpd(self):
        """
        Function to calculate mpd and ms. The mpd=neurons/data that according to
        Vesanto (2000) is 10 x mpd for rough training and 40 x mpd for
        fine-tuning training. However, these factors will be
        considered 20x and 60x to justify data convergence
        missing. These values ​​have not yet been tested and may be
        optimized in the future.

        mpd = number of nodes of the Kohonen map divided by the number of samples of
            input (data lines)
        ms = largest dimension of the Kohonen map
        """

        mn = np.min(self.codebook.mapsize)   # Smallest map size
        max_s = max(self.codebook.mapsize[0], self.codebook.mapsize[1])  # Largest map size

        if mn == 1:
            mpd = float(self.codebook.nnodes * 10) / float(self._dlen)
        else:
            mpd = float(self.codebook.nnodes) / float(self._dlen)
        ms = max_s / 2.0 if mn == 1 else max_s

        return ms, mpd

    def rough_train(self,
                    njob=-1,
                    shared_memory=True,
                    trainlen=None,
                    radiusin=None,
                    radiusfin=None,
                    train_len_factor=1,
                    maxtrainlen=1000):
        """
        Method for implementing the rough training of the SOM object.

        Args:
            njob: number of jobs in parallel.

            shared_memory: shared memory usage.

            trainlen: number of rough training iterations. If not
                filled in, the value defined by:
                20x mpd (neurons/data).

            radiusin: initial rough training radius. In case its not specified
                the value ms/3 will be used (ms-largest dimension of the training
                map).

            radiusfin: initial rough training radius. In case its not specified
                the value radiusin/6 will be used.

            trainlen_factor: factor that will multiply the amount of training
                epochs.

            maxtrainlen: maximum size of allowed iterations for
                training.
        """

        ms, mpd = self._calculate_ms_and_mpd()
        trainlen = min(int(np.ceil(30 * mpd)), maxtrainlen) if not trainlen else trainlen
        trainlen = int(trainlen * train_len_factor)

        # Automatic definition of radius values, in case they have not been defined
        if self.initialization == 'random':
            radiusin = max(1, np.ceil(ms / 3.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin / 4.) if not radiusfin else radiusfin
            self.total_radius = radiusin

        elif self.initialization == 'pca':
            radiusin = max(1, np.ceil(ms / 8.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin / 4.) if not radiusfin else radiusfin
            
        self.train_rough_len = trainlen   
        self.train_rough_radiusin = round(radiusin,2)
        self.train_rough_radiusfin = round(radiusfin,2)
     
        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory)

    def finetune_train(self,
                       njob=-1,
                       shared_memory=True,
                       trainlen=None,
                       radiusin=None,
                       radiusfin=None,
                       train_len_factor=1,
                       maxtrainlen=1000):
        """
        Method for implementing SOM object fine-tuning training.

        Args:
            njob: number of jobs in parallel.

            shared_memory: shared memory usage.

            trainlen: number of iterations of the fine-tuning training. In case
                its not filled in, it will be calculated the value defined by:
                60x mpd (neurons/data).

            radiusin: initial radius of the fine-tuning training. If not
                specified, it will be used the value ms/12 (ms-greatest dimension of the
                training map).

            radiusfin: initial rough training radius. In case its not specified
                the value radiusin/25 will be used.

            trainlen_factor: factor that will multiply the amount of training
                epochs.

            maxtrainlen: maximum size of allowed iterations for
                training.
        """

        ms, mpd = self._calculate_ms_and_mpd()

        # Automatic definition of radius values, in case they have not been defined
        if self.initialization == 'random':
            trainlen = min(int(np.ceil(50 * mpd)), maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, ms / 8.) if not radiusin else radiusin  # of final radius in rough training
            radiusfin = max(1, radiusin / 25.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            trainlen = min(int(np.ceil(40 * mpd)), maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, np.ceil(ms / 8.) / 4) if not radiusin else radiusin
            radiusfin = 1 if not radiusfin else radiusfin  # max(1, ms/128)

        trainlen = int(train_len_factor * trainlen)
    
        self.train_finetune_len = trainlen
        self.train_finetune_radiusin = round(radiusin,2)
        self.train_finetune_radiusfin = round(radiusfin,2)
        
        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory)

    def _batchtrain(self,
                    trainlen,
                    radiusin,
                    radiusfin,
                    njob=-1,
                    shared_memory=True):
        """
        Method for implementing batch training.

        Args:
            trainlen: number of completed training iterations.

            radiusin: initial training radius.

            radiusfin: final training radius.

            njob: number of parallel jobs.

            shared_memory: shared memory usage.

        Returns:
            Returns the result of batch training (updating the
            class variables) for the selected parameters.
        """
        # Find the radius range between the start and end radius with the amount of loop specified by trainlen
        radius = np.linspace(radiusin, radiusfin, trainlen)

        bmu = None

        # Training process for inputs with complete data
        if self.missing == False:     
            if shared_memory:
                data = self.get_data
                data_folder = tempfile.mkdtemp()
                data_name = os.path.join(data_folder, 'data')
                dump(data, data_name)
                data = load(data_name, mmap_mode='r')

            else:
                data = self.get_data
                

            # Training bar
            pbar = tqdm(range(trainlen), mininterval=1)
            for i in pbar:
                if self.bootstrap:
                    # Pass all the data in the last training epoch, to guarantee that
                    # all vectors have BMU
                    if self.actual_train == "Fine" and self.train_finetune_len == i:
                        bootstrap_i = np.arange(0, self._dlen)
                    else:
                        # Create bootstrap indexes for samples from the training array
                        bootstrap_i = np.sort(
                            np.random.choice(
                                np.arange(0, self._dlen, 1), int(self.bootstrap_proportion * self._dlen), 
                                replace=False))
                        
                    # Define the neighborhood for each specified radius
                    neighborhood = self.neighborhood.calculate(
                        self._distance_matrix, radius[i], self.codebook.nnodes)

                    # Find the BMU for the data
                    bmu = self._find_bmu(data[bootstrap_i], njb=njob)

                    # Update the BMU with the data
                    self.codebook.matrix = self._update_codebook_voronoi(data[bootstrap_i],
                                                                         bmu,
                                                                         neighborhood)

                    # X2 is part of the Euclidean distance used during the finding of BMU
                    # for each line of data. As it is a fixed value, it can be ignored during
                    # the encounter of the BMU for each input data, but it is necessary for the calculation of the
                    # quantification error
                    fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(data[bootstrap_i], nan=0.0), np.nan_to_num(data[bootstrap_i], nan=0.0))

                    # Progress bar update
                    pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {round(np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2)),4)}")


                    # Update only the BMU of the vectors that participated in this training epoch
                    self._bmu[:,bootstrap_i] = bmu
                
                # Training without bootstrap
                else:
                    # Define the neighborhood for each specified radius
                    neighborhood = self.neighborhood.calculate(
                        self._distance_matrix, radius[i], self.codebook.nnodes)

                    # Find the BMU for the data
                    bmu = self._find_bmu(data, njb=njob)

                    # Update the BMU with the data
                    self.codebook.matrix = self._update_codebook_voronoi(data,
                                                                        bmu,
                                                                        neighborhood)

                    # X2 is part of the Euclidean distance used during the finding of BMU
                    # for each line of data. As it is a fixed value it can be ignored during
                    # the encounter of the BMU for each input data, but it is necessary for the calculation of the
                    # error quantification
                    fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(data, nan=0.0), np.nan_to_num(data, nan=0.0))

                    # Progress bar update
                    pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {round(np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2)),4)}")
                    
                    # Atualizar os BMUs
                    self._bmu = bmu


        # Training process for inputs with missing data
        elif self.missing == True:
            if shared_memory:
                data = self.get_data
                data_folder = tempfile.mkdtemp()
                data_name = os.path.join(data_folder, 'data')
                dump(data, data_name)
                data = load(data_name, mmap_mode='r+')

            else:
                data = self.get_data


            # Progress bar
            pbar = tqdm(range(trainlen), mininterval=1)
            for i in pbar:
                if self.bootstrap:
                    # Pass all the data in the last training epoch, to guarantee that
                    # all vectors have BMU
                    if i==0:
                        bootstrap_i = np.arange(0, self._dlen)
                    else:
                        # Create bootstrap indexes for samples from the training array
                        bootstrap_i = np.sort(
                            np.random.choice(
                                np.arange(0, self._dlen, 1), int(self.bootstrap_proportion * self._dlen), 
                                replace=False))
                    
                    # Define the neighborhood for each specified radius
                    neighborhood = self.neighborhood.calculate(
                        self._distance_matrix, radius[i], self.codebook.nnodes)

                    # Find the BMU and update the input data for the BMU search
                    # according to training

                    # Display incomplete matrix in rough training
                    if self.actual_train == "Rough":
                        # Find the BMU data
                        bmu = self._find_bmu(data[bootstrap_i], njb=njob)
                        
                        # Update the weights according to the specified neighborhood function
                        self.codebook.matrix = self._update_codebook_voronoi(data[bootstrap_i],
                                                                             bmu,
                                                                             neighborhood, 
                                                                             missing=True)
                        
                        # Fill with the values ​​of previous epochs to keep in case you do not participate
                        # of the current batch
                        data[self.data_missing["indices"]] = self.data_missing["nan_values"]

                        # Fill the empty data locations with the values ​​found in the BMU
                        # every iteration
                        nan_mask = np.isnan(self.data_raw[bootstrap_i])

                        for j in range(self._data[bootstrap_i].shape[0]):
                            bmu_index = bmu[0][j].astype(int)
                            # Insert a randomness and regularization component in the
                            # data imputation during training, so as not to
                            # converge as fast
                            data[bootstrap_i[j]][nan_mask[j]] = self.codebook.matrix[bmu_index][nan_mask[j]]
 
                        # Update missing data
                        self.data_missing["nan_values"] = data[self.data_missing["indices"]]

                        # Delete the data in the data variable
                        data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)

                        if self.save_nan_hist:
                            # Add to nan's processing history
                            self.nan_value_hist.append(self.data_missing["nan_values"])

                        # Progress bar update
                        QE = round(np.mean(np.sqrt(bmu[1])),4)
                        pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {QE}")

                    # Display imputed matrix in fine training
                    elif self.actual_train == "Fine":
                        if self.previous_epoch:
                            # fill in the missing values ​​in data with data from the previous interaction for BMU search
                            data[self.data_missing["indices"]] = self.data_missing["nan_values"]

                            # Find the BMU data
                            bmu = self._find_bmu(data[bootstrap_i], njb=njob)
                            fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(data[bootstrap_i], nan=0.0), np.nan_to_num(data[bootstrap_i], nan=0.0))


                            # Fill the empty data locations with the values ​​found in the BMU
                            # every iteration
                            nan_mask = np.isnan(self.data_raw[bootstrap_i])

                            # Regularization factor
                            reg = radius[i]/self.total_radius-1/self.total_radius

                            for j in range(self._data[bootstrap_i].shape[0]):
                                bmu_index = bmu[0][j].astype(int)
                                # Insert a randomness and regularization component in the
                                # imputation of data during training, so as not to
                                # converge too fast
                                data[bootstrap_i[j]][nan_mask[j]] = self.codebook.matrix[bmu_index][nan_mask[j]]*np.random.uniform(1-reg, 1+reg, np.sum(nan_mask[j]))

                            # Update the weights according to the specified neighborhood function
                            self.codebook.matrix = self._update_codebook_voronoi(data[bootstrap_i],
                                                                                bmu,
                                                                                neighborhood)

                            # Update missing data
                            self.data_missing["nan_values"] = data[self.data_missing["indices"]]

                            # Delete the data in the data variable
                            data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)

                            if self.save_nan_hist:
                                # Add to nan's processing history
                                self.nan_value_hist.append(self.data_missing["nan_values"])

                            # Progress bar update
                            QE = round(np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2)),4)
                            pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {QE}. Reg:{round(reg,2)}")
                        else:
                            # Find the BMU data
                            bmu = self._find_bmu(data[bootstrap_i], njb=njob)
                            
                            # Update the weights according to the specified neighborhood function
                            self.codebook.matrix = self._update_codebook_voronoi(data[bootstrap_i],
                                                                                bmu,
                                                                                neighborhood, 
                                                                                missing=True)
                            
                            # Fill with the values ​​of previous epochs to keep in case you do not participate
                            # of the current batch
                            data[self.data_missing["indices"]] = self.data_missing["nan_values"]

                            # Fill the empty data locations with the values ​​found in the BMU
                            # every iteration
                            nan_mask = np.isnan(self.data_raw[bootstrap_i])

                            for j in range(self._data[bootstrap_i].shape[0]):
                                bmu_index = bmu[0][j].astype(int)
                                # Insert a randomness and regularization component in the
                                # data imputation during training, so as not to
                                # converge as fast
                                data[bootstrap_i[j]][nan_mask[j]] = self.codebook.matrix[bmu_index][nan_mask[j]]
    
                            # Update missing data
                            self.data_missing["nan_values"] = data[self.data_missing["indices"]]

                            # Delete the data in the data variable
                            data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)

                            if self.save_nan_hist:
                                # Add to nan's processing history
                                self.nan_value_hist.append(self.data_missing["nan_values"])

                            # Progress bar update
                            QE = round(np.mean(np.sqrt(bmu[1])),4)
                            pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {QE}")

                    if self.history_plot:
                        if i%2 == 0:
                            self.plot_umatrix(figsize = (5,3),
                                         hits = True,
                                         save = True,
                                         file_name = f"{self.actual_train}_epoch{i+1}",
                                         bmu=bmu)

                    # Update only the BMU of the vectors that participated in this training epoch
                    self._bmu[:, bootstrap_i] = bmu
                else:
                    # Define the neighborhood for each specified radius
                    neighborhood = self.neighborhood.calculate(
                        self._distance_matrix, radius[i], self.codebook.nnodes)

                    # Find the BMU and update the input data for the BMU search
                    # according to training

                    # Display incomplete matrix in rough training
                    if self.actual_train == "Rough":
                        # Find the BMU data
                        bmu = self._find_bmu(data, njb=njob)

                        # Update the weights according to the specified neighborhood function
                        self.codebook.matrix = self._update_codebook_voronoi(data,
                                                                            bmu,
                                                                            neighborhood, 
                                                                            missing=True)
                        
                        # Fill the empty data locations with the values ​​found in the BMU
                        # every iteration
                        nan_mask = np.isnan(self.data_raw)

                        for j in range(self._data.shape[0]):
                            bmu_index = bmu[0][j].astype(int)
                            # Insert a randomness and regularization component in the
                            # data imputation during training, so as not to
                            # converge as fast
                            data[j][nan_mask[j]] = self.codebook.matrix[bmu_index][nan_mask[j]]
                        
                        # Update missing data
                        self.data_missing["nan_values"] = data[self.data_missing["indices"]]

                        # Delete the data in the data variable
                        data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)
                        if self.save_nan_hist:
                            # Add to nan's processing history
                            self.nan_value_hist.append(self.data_missing["nan_values"])

                        # Progress bar update
                        QE = round(np.mean(np.sqrt(bmu[1])),4)
                        pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {QE}")

                    # Display matrix inputed in fine training
                    elif self.actual_train == "Fine":
                        if self.previous_epoch:

                            # Fill in the missing values ​​in data with data from the previous interaction for BMU search
                            data[self.data_missing["indices"]] = self.data_missing["nan_values"]

                            # Find the BMU data
                            bmu = self._find_bmu(data, njb=njob)
                            fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(data, nan=0.0), np.nan_to_num(data, nan=0.0))


                            # Fill the empty data locations with the values ​​found in the BMU
                            # every iteration
                            nan_mask = np.isnan(self.data_raw)

                            # Regularization factor
                            reg = radius[i]/self.total_radius-1/self.total_radius

                            for j in range(self._data.shape[0]):
                                bmu_index = bmu[0][j].astype(int)
                                # Insert a randomness and regularization component in the
                                # data imputation during training, so as not to
                                # converge as fast
                                data[j][nan_mask[j]] = self.codebook.matrix[bmu_index][nan_mask[j]]*np.random.uniform(1-reg, 1+reg, np.sum(nan_mask[j]))
                            
                            # Update the weights according to the specified neighborhood function
                            self.codebook.matrix = self._update_codebook_voronoi(data,
                                                                                bmu,
                                                                                neighborhood)

                            # Update missing data
                            self.data_missing["nan_values"] = data[self.data_missing["indices"]]

                            # Delete the data in the data variable
                            data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)

                            if self.save_nan_hist:
                                # Add to nan's processing history
                                self.nan_value_hist.append(self.data_missing["nan_values"])

                            # Progress bar update
                            QE = round(np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2)),4)
                            pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {QE}. Reg:{round(reg,2)}")
                        else:
                            # Find the BMU data
                            bmu = self._find_bmu(data, njb=njob)

                            # Update the weights according to the specified neighborhood function
                            self.codebook.matrix = self._update_codebook_voronoi(data,
                                                                                bmu,
                                                                                neighborhood, 
                                                                                missing=True)
                            
                            # Fill the empty data locations with the values ​​found in the BMU
                            # every iteration
                            nan_mask = np.isnan(self.data_raw)

                            for j in range(self._data.shape[0]):
                                bmu_index = bmu[0][j].astype(int)
                                # Insert a randomness and regularization component in the
                                # data imputation during training, so as not to
                                # converge as fast
                                data[j][nan_mask[j]] = self.codebook.matrix[bmu_index][nan_mask[j]]
                            
                            # Update missing data
                            self.data_missing["nan_values"] = data[self.data_missing["indices"]]

                            # Delete the data in the data variable
                            data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)
                            if self.save_nan_hist:
                                # Add to nan's processing history
                                self.nan_value_hist.append(self.data_missing["nan_values"])

                            # Progress bar update
                            QE = round(np.mean(np.sqrt(bmu[1])),4)
                            pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {QE}")

                    if self.history_plot:
                        if i%2 == 0:
                            self.plot_umatrix(figsize = (5,3),
                                         hits = True,
                                         save = True,
                                         file_name = f"{self.actual_train}_epoca{i+1}",
                                         bmu=bmu)
                    # Update BMU
                    self._bmu = bmu
            
            self.params_json


    def _find_bmu(self,
                 input_matrix,
                 njb=-1,
                 nth=1,
                 project=False):
        """
        Finds the BMU (Best Matching Units) for each input data through
        of the input data matrix. In a unified way parallelizing the
        calculation instead of going data by data and comparing with the codebook.

        Args:
            input_matrix: numpy ndarray matrix representing the samples of the
                input as rows and variables as columns.

            njb: number of jobs for the parallel search. Default: -1 (automatic
                search by the number of processing colors)

            nth:

            project: boolean value for matching BMU related to a
                projection base on an already trained map.

        Returns:
            The BMU for each input data in the format [[bmus],[distances]].
        """

        dlen = input_matrix.shape[0]
        if njb == -1:
            njb = cpu_count()
        y2 = np.einsum('ij,ij->i', self.codebook.matrix, self.codebook.matrix)
        
        
        pool = Pool(njb)

        # Create object to find BMU in pieces of data
        chunk_bmu_finder = self._chunk_based_bmu_find

        def row_chunk(part):
            return part * dlen // njb

        def col_chunk(part):
            return min((part + 1) * dlen // njb, dlen)
        
        # Separate chunks of the input input_matrix data to be parsed
        # by chunk_bmu_finder
        chunks = [input_matrix[row_chunk(i):col_chunk(i)] for i in range(njb)]


        if project:
            missing_proj = np.isnan(input_matrix).any()
            if missing_proj:
                # Map the data chunks and apply the chunk_bmu_finder method on each chunk, finding the BMU for each chunk
                b = pool.map(lambda chk: chunk_bmu_finder(chk,
                                                          self.codebook.matrix[:, :input_matrix.shape[1]],
                                                          y2,
                                                          nth=nth, 
                                                          project=project,
                                                          missing=missing_proj),
                                                          chunks)
            else:
                # Map the data chunks and apply the chunk_bmu_finder method on each chunk, finding the BMU for each chunk
                b = pool.map(lambda chk: chunk_bmu_finder(chk,
                                    self.codebook.matrix[:, :input_matrix.shape[1]], 
                                    y2=y2,
                                    project=project,
                                    nth=nth),
                             chunks)
                
        else:
            if self.missing:
                # Map the data chunks and apply the chunk_bmu_finder method on each chunk, finding the BMU for each chunk
                b = pool.map(lambda chk: chunk_bmu_finder(chk,
                                                          self.codebook.matrix,
                                                          y2=y2,
                                                          nth=nth, 
                                                          missing=True),
                                                          chunks)
            else:
                # Map the data chunks and apply the chunk_bmu_finder method on each chunk, finding the BMU for each chunk
                b = pool.map(lambda chk: chunk_bmu_finder(chk,
                                                          self.codebook.matrix, 
                                                          y2,
                                                          nth=nth),
                                                          chunks)
        pool.close()
        pool.join()

        # Arrange the BMU data chunks into an array [2,dlen] where the
        # first line has the BMU and the second the distances
        bmu = np.asarray(list(itertools.chain(*b))).T
        del b
        return bmu


    def _update_codebook_voronoi(self, training_data, bmu, neighborhood, missing=False):
        """
        Method to update the weights of each node in the codebook that belongs to
        neighborhood of the BMU. First find each node's Voronoi set. It needs to
        calculate a smaller matrix. Faster than the classic algorithm in
        batch, it is based on the implementation of the SOM Toolbox algorithm for
        MATLAB by the University of Helsinky. First implemented in
        Python by the SOMPY library.

        Args:
            training_data: array of input vectors as rows and variables
                as columns.

            bmu: BMU for each input data. has the format
                [[bmus],[distances]].

            neighborhood: matrix representing the neighborhood of each BMU.

        Returns:
            An updated codebook that incorporates learning from the input data.
        """
        if missing:
            # Create a mask for the missing values ​​in training_data and replace it with 0
           training_data[np.isnan(training_data)] = 0

        # Get all the BMU numbers from each data line and put them in the
        # int format
        row = bmu[0].astype(int)

        # All indexes for columns
        col = np.arange(training_data.shape[0])

        # Array with 1 repeated in the length of the lines of data
        val = np.tile(1, training_data.shape[0])

        # Create a sparse matrix (csr -> compressed sparsed row) with the call
        # csr_matrix((val, (row, col)), [shape=(nnodes, dlen)])
        P = csr_matrix((val, (row, col)), shape=(self.codebook.nnodes,
                                                 training_data.shape[0]))

        # Multiply by the input data to return a matrix S with the
        # input data in BMU
        S = P.dot(training_data)

        # Multiply the neighborhood values ​​by the matrix S with the values ​​of
        # input in the BMU
        nom = neighborhood.T.dot(S)

        # Count how many times each BMU was selected by an input vector
        nV = P.sum(axis=1).reshape(1, self.codebook.nnodes)

        # Multiply the amount of times the BMU was selected by the
        # values ​​input by the neighborhood function
        denom = nV.dot(neighborhood.T).reshape(self.codebook.nnodes, 1)

        # Divide the values ​​in the nominator by the denominator
        new_codebook = np.divide(nom, denom)

        return np.around(new_codebook, decimals=6)
    
    def _chunk_based_bmu_find(self,
                            input_matrix, 
                            codebook, 
                            y2=None, 
                            nth=1, 
                            missing=False,
                            project=False):
        """
        Finds the BMU corresponding to the input data matrix.

        Args:
            input_matrix: a matrix of the input data, representing the input
                vectors in the rows and the vector variables in the columns. When the
                search is parallelized, the input matrix can be a sub-matrix of
                a larger array.

            codebook: matrix of weights to be used in the search for BMU.

            nth:

        Returns:
            Returns the BMU and distances for the matrix or sub-matrix of vectors
            input. In the format [[bmu],[distance]].
        """
        def dist_by_type(codebook, ddata, missing, train_type=None):
            """
            Function to choose the type of distance to be calculated depending on the presence of data
            absences and/or the training stage.
            """
            if missing:
                if train_type == "nan_euclid" or train_type == "Projected":
                    d = nan_euclidean_distances(codebook, ddata)
                else:
                    d = np.dot(codebook, ddata.T)
                    d *= -2
                    d += y2.reshape(nnodes, 1)
            else:
                d = np.dot(codebook, ddata.T)
                d *= -2
                d += y2.reshape(nnodes, 1)
            
            return d

        # Number of entries (rows) of the input data
        dlen = input_matrix.shape[0]

        # Initialize an array of dlen rows and two columns
        bmu = np.empty((dlen, 2))
        
        # Number of training nodes in the codebook
        nnodes = codebook.shape[0]

        # Batch size
        blen = min(50, dlen)

        # While loop initializer
        i0 = 0
        
        if missing:
            while i0 + 1 <= dlen:
                # Start searching the data (rows of the input matrix)
                low = i0  

                # End of data search (rows of the input matrix) for this batch
                high = min(dlen, i0 + blen)
                i0 = i0 + blen  # Update loop initializer

                # Clipping on the input matrix in these batch samples
                ddata = input_matrix[low:high + 1]

                if self.actual_train == "Rough" or self.previous_epoch == False:
                    type_search = "nan_euclid"
                else:
                    type_search = "Fine"

                d = dist_by_type(codebook=codebook, 
                                 ddata=ddata, 
                                 missing=missing, 
                                 train_type = "Projected" if project else type_search)

                # Find the BMU
                # Function to find the position within the distances array in which is
                # the smallest value (the BMU), and designate it as the first attribute of the variable
                # BMU
                bmu[low:high + 1, 0] = np.argpartition(d, nth, axis=0)[nth - 1]

                # Function to get the smallest distance and designate as second attribute of
                # BMU variable
                bmu[low:high + 1, 1] = np.partition(d, nth, axis=0)[nth - 1]
                del ddata
        else:
            while i0 + 1 <= dlen:
                # Start searching the data (rows of the input matrix)
                low = i0  

                # End of data search (rows of the input matrix) for this batch
                high = min(dlen, i0 + blen)
                i0 = i0 + blen  # Update loop initializer

                # Clipping on the input matrix in these batch samples
                ddata = input_matrix[low:high + 1]

                d = dist_by_type(codebook=codebook, 
                                 ddata=ddata, 
                                 missing=missing)

                # Find the BMU
                # Function to find the position within the distances array in which is
                # the smallest value (the BMU), and designate it as the first attribute of the variable
                # BMU
                bmu[low:high + 1, 0] = np.argpartition(d, nth, axis=0)[nth - 1]

                # Function to get the smallest distance and designate as second attribute of
                # BMU variable
                bmu[low:high + 1, 1] = np.partition(d, nth, axis=0)[nth - 1]
                del ddata

        return bmu
    
    @property
    def topographic_error(self):
        """
        Function that is in SOMPY, I disagree with this function, it searches only if the first and second BMU that
        best represent the input vector are neighbors.
        """
        bmus2 = self._find_bmu(self.get_data, nth=2)
               
        dist_matrix_1 = self._distance_matrix
        topographic_error = 1-(np.array(
                [distances[bmu2] for bmu2, distances in zip(bmus2[0].astype(int), dist_matrix_1)]) > 4).mean()

        return topographic_error
    
    @property
    def calculate_topographic_error(self):
        """
        Calculation of the Topographic Error, which is a measure of the quality of the
        preservation of the space structure of the input data in the trained
        map. It is calculated by finding the smallest and second smallest neurons
        Euclidean distance of each input vector and evaluating whether the
        preservation of this neighborhood takes place in the trained output map. If
        the preservation of the neighborhood occurs so it is said that there was preservation of the
        topology and the topographic error is low.
        For each input vector that is not neighborhood preserved it is
        counted in proportion to the topographical error, so it is an error that varies
        from 0 (all input vectors maintained the neighborhood) to 1 (none
        input vector preserved its neighborhood).
        """

        # Search in chunks to avoid a RAM bottleneck
        # Start the while loop
        i0 = 0
        
        # Batch size
        blen = 1000
        
        # Indexes
        indices_before = np.zeros((self.data_raw.shape[0],2))
        indices_after = np.zeros((self.data_raw.shape[0],2))
        
        while i0+1 <= self._dlen:
            # Start data search
            low = i0

            # End data search
            high = min(self._dlen, i0+blen)

            # Cut the matrix for this batch
            ddata = self.data_raw[low:high + 1]

            # Pre-training distance matrix
            dist_before = nan_euclidean_distances(ddata, ddata)

            # Find vectors closest to input vectors
            argmin_before = np.argsort(dist_before, axis=0)[1,:]

            # Paired indexes of closest vectors
            indices_before_batch = np.array([[i,j] for i,j in zip(np.arange(0, len(ddata), 1), argmin_before)])

            # Fill main vector
            for i, j in zip(np.arange(low, high+1, 1), np.arange(0, blen+1, 1)):
                indices_before[i] = indices_before_batch[j]            

            # Generate cubic coordinates for the map
            coordinates = self._generate_oddr_cube_lattice(self.mapsize[0], self.mapsize[1])
            cols, rows = self.mapsize[0], self.mapsize[1]
            
            # Create toroidal neighborhood
            toroid_neigh = [[0, 0], [cols, 0], [cols, rows], [0, rows], [-cols, rows], [-cols, 0], [-cols, -rows], [0,-rows], [cols, -rows]]
            toroid_neigh = [self._oddr_to_cube(i[0], i[1]) for i in toroid_neigh]
            
            # Capture BMU
            bmus = self._bmu[0][low:high + 1].astype(int)
            
            # Create empty matrix to fill the Manhatan distances inside a hexagonal cubic grid
            dist_after = np.zeros([len(ddata),len(ddata)])
            for i in range(len(ddata)):
                for j in range(len(ddata)):
                    dist = int(min([self._cube_distance(coordinates[bmus[i]] + neig,coordinates[bmus[j]]) for neig in toroid_neigh]))
                    dist_after[j][i] = dist
                    
            # Find closest hits to BMU
            argmin_after = np.argsort(dist_after, axis=0)[1,:]
            
            # Paired indexes of these hits
            indices_after_batch = np.array([[i,j] for i,j in zip(np.arange(0, len(ddata), 1), argmin_after)])
            
            # Fill main vector
            for i, j in zip(np.arange(low, high+1, 1), np.arange(0, blen+1, 1)):
                indices_after[i] = indices_after_batch[i]
            
            # Update loop index
            i0=i0+blen
            
            del ddata

        # Topographic error: 0 if the neighborhood is maintained and 1 if not
        topo_error = sum([0 if i==j else 1 for i,j in zip(indices_before, indices_after)])/len(self.data_raw)

        return 1-topo_error

    @property
    def calculate_quantization_error(self):
        data = self.get_data
        data[self.data_missing["indices"]] = self.data_missing["nan_values"]
        fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(self.get_data, nan=0.0), np.nan_to_num(self.get_data, nan=0.0))
        return np.mean(np.sqrt(np.abs(self._bmu[1] + fixed_euclidean_x2)))
    
    @property
    def calculate_quantization_error_expanded(self):
        data = self.get_data
        data[self.data_missing["indices"]] = self.data_missing["nan_values"]
        fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(self.get_data, nan=0.0), np.nan_to_num(self.get_data, nan=0.0))
        return np.sqrt(np.abs(self._bmu[1] + fixed_euclidean_x2))
        

    def build_umatrix(self, expanded=False):
        """
        Function to calculate the U-Matrix of unified distances from the
        trained weight matrix.

        Args:
            expanded: boolean value to indicate whether the return will be from the
                summarized unified distances matrix (average of distances from the 6
                neighborhood BMU) or expanded (all distance values)
                
        Returns:
            Expanded or summarized unified distances matrix.
        """
        # Function to find distance quickly
        def fast_norm(x):
            """
            Retorna a norma L2 de um array 1-D.
            """
            return np.sqrt(np.dot(x, x.T))

        # Matrix of BMU weights
        weights = np.reshape(self.codebook.matrix, (self.mapsize[1], self.mapsize[0], self.codebook.matrix.shape[1]))

        # Neighbor hexagonal search
        ii = [[1, 1, 0, -1, 0, 1], [1, 0,-1, -1, -1, 0]]
        jj = [[0, 1, 1, 0, -1, -1], [0, 1, 1, 0, -1, -1]]

        # Initialize U-Matrix
        um = np.nan * np.zeros((weights.shape[0], weights.shape[1], 6))

        # Fill U-Matrix
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                w_2 = weights[y, x]
                e = y % 2 == 0
                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                    if (x+i >= 0 and x+i < weights.shape[1] and y+j >= 0 and y+j < weights.shape[0]):
                        w_1 = weights[y+j, x+i]
                        um[y, x, k] = fast_norm(w_2-w_1)
        if expanded:
            # Expanded U-Matrix
            return um
        else:
            # Reduced U-Matrix
            return np.nanmean(um, axis=2)
        
    
    def plot_umatrix(self,
                     figsize = (10,10),
                     hits = True,
                     save = True,
                     file_name = None,
                     file_path = False,
                     bmu=None):
        
        if file_name is None:
            file_name = f"U_matrix_{self.name}"

        if hits:
            # Hit count
            unique, counts = np.unique(bmu[0].astype(int), return_counts=True)

            # Normalize this count from 0.5 to 2.0 (from a small hexagon to a
            # hexagon that covers half of the neighbors).
            counts = minmax_scale(counts, feature_range = (0.5,2))

            bmu_dic = dict(zip(unique, counts))
            
        
        # Neighbor hexagonal search
        ii = [[1, 1, 0, -1, 0, 1], [1, 0,-1, -1, -1, 0]]
        jj = [[0, 1, 1, 0, -1, -1], [0, 1, 1, 0, -1, -1]]
        
        # U-Matrix
        xx = np.reshape(self._generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,0], (self.mapsize[1], self.mapsize[0]))
        yy = np.reshape(self._generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,1], (self.mapsize[1], self.mapsize[0]))
        
        # Plotting
        um = self.build_umatrix(expanded = True)
        umat = self.build_umatrix(expanded = False)
        
        # Plotagem
        f = plt.figure(figsize=figsize)
        ax = f.add_subplot()
        ax.set_aspect('equal')
        
         # Normalize colors for all hexagons
        norm = mpl.colors.Normalize(vmin=np.nanmin(um), vmax=np.nanmax(um))
        counter = 0
        
        for j in range(self.mapsize[1]):
            for i in range(self.mapsize[0]):
                # Central Hexagon
                hex = RegularPolygon((xx[(j,i)]*2,
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
                    
                # Plot hits
                if hits:
                    try:
                        hex = RegularPolygon((xx[(j, i)]*2,
                                              yy[(j,i)]*2),
                                             numVertices=6,
                                             radius=((1/np.sqrt(3))*bmu_dic[counter]),
                                             facecolor='white',
                                             alpha=1)
                        ax.add_patch(hex)
                    except:
                        pass

                counter+=1
                
        plt.xlim(-0.5, 2*self.mapsize[0]-0.5)
        plt.ylim(-0.5660254, 2*self.mapsize[1]*0.8660254-2*0.560254)
        plt.tight_layout()
        ax.set_axis_off()
        plt.gca().invert_yaxis()
        plt.close()
        
        if save:
            if file_path:
                f.savefig(f"{file_path}/{file_name}.jpg",dpi=300, bbox_inches = "tight")
            else:
                # Create directories if they don't exist
                path = 'Plots/U_matrix'
                os.makedirs(path, exist_ok=True)
                if hits:
                    f.savefig(f"Plots/U_matrix/{file_name}_with_hits.jpg",dpi=300, bbox_inches = "tight")
                else:
                    f.savefig(f"Plots/U_matrix/{file_name}.jpg",dpi=300, bbox_inches = "tight")
    
    def rep_sample(self, save=False, project=None):
        """
        Returns a dictionary containing the representative samples for each best-matching neuron
        (BMU) of the self-organizing map (SOM).

        Args: 
            save (bool, optional): indicates whether the results should be saved in a text file. 
            Default is False.

        Returns:
            dict: a dictionary in which the keys are the BMU and the values are the representative samples 
            associated to each BMU, in order of representativeness.
        """
        if project is not None:
            som_bmus = np.concatenate((self._bmu[0].astype(int),np.array(project.BMU.values-1)))
            sample_names = np.concatenate((np.array(self._sample_names), np.array(project.index.values)))
            data = np.concatenate((self.get_data, self.data_proj_norm), axis=0)
        else:
            som_bmus = self._bmu[0].astype(int)
            sample_names = self._sample_names
            data = self.get_data

        # Dictionary of labels with samples
        dic = {}
        for key, value in zip(som_bmus, sample_names):
            if key in dic:
                if isinstance(dic[key], list):
                    dic[key].append(value)
                else:
                    dic[key] = [dic[key], value]
            else:
                dic[key] = value

        # Dictionary of sample indexes in each BMU
        dic_index = {}
        for key, index in zip(som_bmus, range(len(sample_names))):
            if key in dic_index:
                if isinstance(dic_index[key], list):
                    dic_index[key].append(index)
                else:
                    dic_index[key] = [dic_index[key], index]
            else:
                dic_index[key] = index
        
        # Reorganize the dictionary by order of distances
        rep_samples_dic = {}
        for bmu, bmu in zip(dic, dic_index):
            samples_name = dic[bmu]
            samples_index = dic_index[bmu]

            if isinstance(samples_name, list):
                bmu_vector = self.codebook.matrix[bmu].reshape(1,-1)
                data_vectors = data[samples_index]
                dist_mat = nan_euclidean_distances(bmu_vector, data_vectors)
                sorted_ind = np.argsort(dist_mat).ravel()
                rep_samples = list(np.array(samples_name)[sorted_ind])
            else:
                rep_samples = samples_name
            
            rep_samples_dic[bmu] = rep_samples
        
        if save:
            name = "Projected_representative_samples" if project is not None else "Representative_samples"
            with open(f'Results/{name}.txt', 'w', encoding='utf-8') as file:
                for key, value in rep_samples_dic.items():
                    if isinstance(value, list):
                        value = ', '.join(value)
                    file.write(f'BMU {key+1}: {value}\n')

        return rep_samples_dic


    def _expected_mapsize(self, data):
        """
        Returns the expected size of the map based on the heuristic function defined by
        Vesanto et al (2000) defined by: 5 x sqrt(M).

        Args:
            data: the input data for the SOM training.

        """
        expected = round(np.sqrt(5*np.sqrt(data.shape[0])))

        if expected%2!=0:
            row_expec = expected+1
        else:
            row_expec = expected

        return (expected, row_expec)  


    def _generate_hex_lattice(self, n_columns, n_rows):
        """
        Generates the xy coordinates of the BMU for an odd-r hexagonal grid (odd
        columns shifted to the right).

        Args:
            n_rows: number of lines in the Kohonen map.

            n_columns: number of columns in the Kohonen map.

        Returns:
            Coordinates in the [x,y] format for the BMU in a hexagonal grid.

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


    def _generate_rec_lattice(self, n_columns, n_rows):
        """
        Generates the xy coordinates of the BMU for a rectangular grid.

        Args:
            n_rows: number of rows in the Kohonen map.
            n_columns: number of columns in the Kohonen map.

        returns:
            Coordinates in the [x,y] format for the BMU in a rectangular grid.

        """
        x_coord = []
        y_coord = []
        for j in range(n_rows):
            for i in range(n_columns):
                x_coord.append(i)
                y_coord.append(j)
        coordinates = np.column_stack([x_coord, y_coord])
        return coordinates


    def _oddr_to_cube(self, col, row):
        """
        Transforms coordinates from rectangular to cubic.

        Args:
            col: column coordinate you want to transform.

            row: coordinate of the row you want to transform.

        Returns:
            Cubic coordinate in [x,y,z] format
        """

        x = col - (row - (row & 1)) / 2
        z = row
        y = -x-z
        return [x, y, z]


    def _cube_distance(self,a, b):
        """
        Calculates the Manhattan distance between two cubic coordinates.
        
        Args:
            a: first cubic coordinate in [x,y,z] format

            b: second cubic coordinate in [x,y,z] format

        Returns:
            Manhattan distance between coordinates.
        """
        return (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])) / 2


    def _generate_oddr_cube_lattice(self, n_columns, n_rows):
        """
        Function to generate cubic coordinates in [x,y,z] format for an odd-r hexagonal
        grid (odd lines shifted to the right) for a
        predetermined number of columns and rows.

        Args:
            n_columns: number of columns.

            n_rows: number of rows.

        Returns:
            coordinates: list[x, y, z]
        """
        x_coord = []
        y_coord = []
        z_coord = []
        for j in range(n_rows):
            for i in range(n_columns):
                x = i-(j-(j & 1))/2
                z = j
                y = -x -z

                # Put in lists
                x_coord.append(int(x))
                y_coord.append(int(y))
                z_coord.append(int(z))

        coordinates = np.column_stack([x_coord, y_coord, z_coord])
        return coordinates




# Silence matplotlib logging
import logging
import sys
logging.getLogger('mtb.font_manager').disabled = True
logging.disable(sys.maxsize)

import warnings

# Ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
