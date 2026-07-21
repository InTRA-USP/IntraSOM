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
              pred_size=0,
              dist_factor = 2, 
              pace_size=500,
              feature_weights=None,
              bmu_compute_dtype='preserve',
              bmu_max_memory_mb=256,
              bmu_config=None):
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
            
            dist_factor: the factor by wich the cube distance will be elevated to
                calculate the map distances, the bigger the value, the faster the
                map will converge. Attention should be addressed in fast convergions
                and lack o convergence dute to small dist_factor. Default:2.

        Returns:
            SOM object with all its inherited methods and attributes.

        """
        # Apply normalization if it is defined
        normalization = "None" if normalization is None else normalization

        if normalization == "var_weighted":

            if feature_weights is None:
                raise ValueError(
                    "feature_weights must be provided when "
                    "normalization='var_weighted'."
                )

            if len(feature_weights) != data.shape[1]:
                raise ValueError(
                    f"Expected {data.shape[1]} feature weights, "
                    f"but received {len(feature_weights)}."
                )

            normalizer = NormalizerFactory.build(
                normalization,
                weights=feature_weights
            )

        else:

            normalizer = NormalizerFactory.build(
                normalization
            )
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
                   pred_size = pred_size,
                   dist_factor = dist_factor,
                   pace_size=pace_size,
                   feature_weights=feature_weights,
                   bmu_compute_dtype=bmu_compute_dtype,
                   bmu_max_memory_mb=bmu_max_memory_mb,
                   bmu_config=bmu_config)

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

        feature_weights = params.get(
            "feature_weights",
            None
        )

        if normalization == "var_weighted":
            normalizer = NormalizerFactory.build(
                normalization,
                weights=feature_weights
            )
        else:
            normalizer = NormalizerFactory.build(normalization)
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
        dist_factor = params["dist_factor"]
        load_param=True
        pace_size = params["pace_size"]
        bmu_compute_dtype = params.get("bmu_compute_dtype", "preserve")
        bmu_max_memory_mb = params.get("bmu_max_memory_mb", 256)
        bmu_config = params.get("bmu_config", None)

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
                   bmus = bmus,
                   dist_factor = dist_factor,
                   pace_size=pace_size,
                   feature_weights=feature_weights,
                   bmu_compute_dtype=bmu_compute_dtype,
                   bmu_max_memory_mb=bmu_max_memory_mb,
                   bmu_config=bmu_config)

class SOM(object):

    def __init__(self,
                 data,
                 neighborhood='gaussian',
                 normalizer="var",
                 normalization = "var",
                 feature_weights=None,
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
                 missing_imput=None,
                 pred_size=0,
                 load_param=False,
                 trained_neurons=None, 
                 bmus = None,
                 dist_factor = 2,
                 pace_size=100_000,
                 bmu_compute_dtype='preserve',
                 bmu_max_memory_mb=256,
                 bmu_config=None):

        # Mask for missing values
        self.mask = mask
        self.pace_size = pace_size

        # BMU configuration. The public arguments expose the settings that are
        # most useful in ordinary workflows. Less common tuning options remain
        # available through bmu_config. Conservative defaults preserve the
        # input precision and cap the temporary score workspace at 256 MiB.
        self.bmu_compute_dtype = str(bmu_compute_dtype).lower()
        if self.bmu_compute_dtype not in {"preserve", "float32", "float64"}:
            raise ValueError(
                "bmu_compute_dtype must be 'preserve', 'float32', or 'float64'."
            )
        self.bmu_max_memory_mb = float(bmu_max_memory_mb)
        if self.bmu_max_memory_mb <= 0:
            raise ValueError("bmu_max_memory_mb must be greater than zero.")

        advanced_defaults = {
            "neuron_block_size": None,
            "memory_safety_factor": 2.0,
            "min_neuron_block_size": 1,
            "force_two_axis_blocking": False,
        }
        if bmu_config is not None:
            if not isinstance(bmu_config, dict):
                raise TypeError("bmu_config must be a dictionary or None.")
            unknown = set(bmu_config) - set(advanced_defaults)
            if unknown:
                raise ValueError(
                    "Unknown bmu_config option(s): " + ", ".join(sorted(unknown))
                )
            advanced_defaults.update(bmu_config)
        if float(advanced_defaults["memory_safety_factor"]) < 1.0:
            raise ValueError("bmu_config['memory_safety_factor'] must be >= 1.")
        if int(advanced_defaults["min_neuron_block_size"]) < 1:
            raise ValueError("bmu_config['min_neuron_block_size'] must be >= 1.")
        self.bmu_config = advanced_defaults

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
        self._normalization = "None" if normalization is None else normalization
        self.feature_weights = feature_weights
        self._dim = data.shape[1]
        self._dlen = data.shape[0]
        self.pred_size = pred_size
        self.name = name
        self.dist_factor = dist_factor
        self.missing = missing
        if self.missing == False:
            if np.isnan(self._data).any():
                sys.exit("Database with missing data, flag 'missing' as True")
            if mask is not None:
                sys.exit("The parameter 'mask' is only used in databases with missing values, flag as None or remove this parameter")
        elif self.missing == True:
            if not np.isnan(self._data).any():
                sys.exit("Database with no missing data, flag 'missing' as False")

        print("Creating neighborhood...")
        self.neighborhood = neighborhood
        self._unit_names = unit_names if unit_names else [f"Unit {var}" for var in self._component_names]
        self.mapshape = mapshape
        self.initialization = initialization
        
        # --------------------------------------------------
        # MAP SIZE
        # Public convention:
        #
        # mapsize = (columns, rows)
        # --------------------------------------------------

        if mapsize is None:

            self.mapsize = self._expected_mapsize(
                self._data
            )

        elif isinstance(
            mapsize,
            (int, np.integer)
        ):

            # Integer mapsize is interpreted as the total
            # desired number of neurons.
            n_nodes = int(mapsize)

            if n_nodes <= 0:
                raise ValueError(
                    "mapsize must contain a positive "
                    "number of neurons."
                )

            # --------------------------------------------------
            # Find all exact factor pairs:
            #
            # n_nodes = columns * rows
            #
            # Preference is given to the most square-like map.
            # --------------------------------------------------

            candidates = []

            max_factor = int(
                np.sqrt(n_nodes)
            )

            for factor in range(
                1,
                max_factor + 1
            ):

                if n_nodes % factor == 0:

                    other = (
                        n_nodes // factor
                    )

                    # mapsize convention:
                    # (columns, rows)

                    candidates.append(
                        (
                            other,
                            factor
                        )
                    )

                    if other != factor:

                        candidates.append(
                            (
                                factor,
                                other
                            )
                        )

            # --------------------------------------------------
            # Hexagonal toroidal maps require an even
            # number of rows.
            # --------------------------------------------------

            if (
                lattice == "hexa"
                and
                mapshape == "toroid"
            ):

                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[1] % 2 == 0
                ]

                if not candidates:

                    raise ValueError(
                        f"Cannot create an exact "
                        f"hexagonal toroidal map with "
                        f"{n_nodes} neurons and an even "
                        f"number of rows. "
                        f"Provide mapsize=(columns, rows) "
                        f"explicitly."
                    )

            # Choose the factor pair closest to a square
            self.mapsize = min(
                candidates,
                key=lambda size: abs(
                    size[0] - size[1]
                )
            )

        elif isinstance(
            mapsize,
            (tuple, list, np.ndarray)
        ):

            if len(mapsize) != 2:

                raise ValueError(
                    "mapsize must contain exactly "
                    "two values in the format "
                    "(columns, rows)."
                )

            cols = int(
                mapsize[0]
            )

            rows = int(
                mapsize[1]
            )

            if cols <= 0 or rows <= 0:

                raise ValueError(
                    "The number of columns and rows "
                    "must be greater than zero."
                )

            # Only hexagonal toroidal maps require
            # an even number of rows.
            if (
                lattice == "hexa"
                and
                mapshape == "toroid"
                and
                rows % 2 != 0
            ):

                rows += 1

                print(
                    "Hexagonal toroidal maps require "
                    "an even number of rows. "
                    f"The map size has been changed to "
                    f"({cols}, {rows})."
                )

            self.mapsize = (
                cols,
                rows
            )

        else:

            raise TypeError(
                "mapsize must be None, an integer, "
                "or a tuple/list in the format "
                "(columns, rows)."
            )


        self.QE = 0
        self.QE_expanded = np.zeros(self._dlen)

        self.lattice = lattice
        self.training = training
        self.load_param = load_param
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
            self.codebook = Codebook(self.mapsize, self.lattice, self.mapshape, self.dist_factor)
            self.codebook.matrix = self._normalizer.normalize_by(self.data_raw, trained_neurons.iloc[:,7:].values)
            
            print("Loading distances matrix...")
            self._distance_matrix = self.calculate_map_dist
        else:
            self.data_missing = {"indices":tuple(zip(*np.argwhere(np.isnan(self.data_raw)))), 
                                 "nan_values":None}
            self._bmu = np.zeros((2,self._dlen))
            self.codebook = Codebook(self.mapsize, self.lattice, self.mapshape, self.dist_factor)
            print("Loading distances matrix...")
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
        return self._data.copy()
    
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
            elif isinstance(obj, tuple):
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
        dic["feature_weights"] = (
                None
                if self.feature_weights is None
                else list(self.feature_weights)
            )
        dic["initialization"] = self.initialization
        dic["training"] = self.training
        dic["pace_size"] = self.pace_size
        dic["bmu_compute_dtype"] = self.bmu_compute_dtype
        dic["bmu_max_memory_mb"] = self.bmu_max_memory_mb
        dic["bmu_config"] = dict(self.bmu_config)
        dic["name"] = self.name
        dic["component_names"] = list(self._component_names)
        dic["unit_names"] = list(self._unit_names)
        dic["sample_names"] = list(self._sample_names)
        dic["missing"] = self.missing
        dic["pred_size"] = int(self.pred_size)
        dic["dist_factor"] = self.dist_factor
        dic["bmus"] = self._bmu[0].astype(int).tolist()
        if self.missing == True:
            dic["bmus_dist"] = list(self.QE_expanded)
            dic["missing_imput"] = list(self.data_missing["nan_values"])
        elif self.missing == False:
            dic["bmus_dist"] = list(self.QE_expanded)
            
        
        # Fix serialization problems
        dic = fix_serialize(dic)

        # Transform to JSON
        json_params = json.dumps(dic)

        # Save the result into the specified directory
        path = 'Results'
        os.makedirs(path, exist_ok=True)
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

        nnodes = self.codebook.nnodes

        distance_matrix = np.zeros(
            (nnodes, nnodes),
            dtype=float
        )

        print("Initializing map...")

        # --------------------------------------------------
        # TOROIDAL
        # --------------------------------------------------

        if self.mapshape == "toroid":

            # Only the distances from neuron 0 are calculated
            initial_matrix = (
                self.codebook
                .grid_dist(0)
                .reshape(
                    self.mapsize[1],  # rows
                    self.mapsize[0]   # columns
                )
            )

            counter = 0

            for i in tqdm(
                range(self.mapsize[1]),
                position=0,
                leave=True,
                desc="Creating Neuron Distance Rows",
                unit="rows"
            ):

                for j in range(self.mapsize[0]):

                    # Periodic translation
                    shifted = np.roll(
                        initial_matrix,
                        shift=(i, j),
                        axis=(0, 1)
                    )

                    # Extra correction only for odd-r
                    # HEXAGONAL lattices
                    if (
                        self.lattice == "hexa"
                        and i % 2 != 0
                    ):

                        shifted[0::2] = np.roll(
                            shifted[0::2],
                            shift=1,
                            axis=1
                        )

                    distance_matrix[counter] = (
                        shifted.ravel()
                    )

                    counter += 1

        # --------------------------------------------------
        # PLANAR
        # --------------------------------------------------

        elif self.mapshape == "planar":

            for i in tqdm(
                range(nnodes),
                desc="Creating Neuron Distance Rows",
                unit="Neurons"
            ):

                distance_matrix[i, :] = (
                    self.codebook.grid_dist(i)
                )

        else:

            raise ValueError(
                "mapshape only accepts "
                "'toroid' or 'planar'."
            )

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
        QE = self.QE_expanded
        results_df.insert(1, "q-error", QE)

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
        dic_params = {
            "var": "Variance",
            "var_weighted": "Weighted Variance",
            "None": "None",
            "toroid": "Toroid",
            "hexa": "Hexagonal",
            "random": "Randomic",
            "gaussian": "Gaussian",
            True: "Yes"
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
        if self._normalizer.name == "var_weighted":

            text_file.write("Feature Weights:\n")

            for feature, weight in zip(
                self._component_names,
                self.feature_weights
            ):
                text_file.write(
                    f"  {feature}: {weight}\n"
                )
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
        text_file.write(f"Quantization Error: {round(self.QE,4)}\n")
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
                        save = True,
                        imput = False):
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
        if imput:
            original = data_proj.values

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
            c_shape = data_proj.shape[1]
            data_proj = self._normalizer.normalize_by(self.data_raw[:,:c_shape],
                                                      data_proj,
                                                      with_labels=False,
                                                      pred_size=self.pred_size)

        self.data_proj_norm = data_proj

        # Find the BMU for this new data
        bmus = self._find_bmu(data_proj, project=True, pace_size=self.pace_size)
        bmus_ind = bmus[0].astype(int)

        # Rescue the BMU dataframe
        bmu_df = self.neurons_dataframe

        # Create dataframe for projection
        projected_df = bmu_df.iloc[bmus_ind,:]

        if imput:
            # Transform to arrays
            projected = projected_df.iloc[:,7:].values

            # Pad original data with nan in case of imputing missing features
            if projected.shape != original.shape:
                shape_diff = np.subtract(projected.shape, original.shape)
                padding = [(0, shape_diff[i]) for i in range(len(shape_diff))]
                original = np.pad(original, padding, mode='constant', constant_values=np.nan)
            
            # Replace nans by trained values
            nan_mask = np.isnan(original)
            original[nan_mask] = projected[nan_mask]

            # Recreate dataframe
            columns = [x[2:] for x in projected_df.iloc[:,7:].columns]
            index = projected_df.index.values
            projected_df = pd.DataFrame(original,
                                        columns=columns, 
                                        index=index)

        else:
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
              previous_epoch = True):
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
                    bmu = self._find_bmu(data[bootstrap_i], njb=njob, pace_size=self.pace_size)

                    # Update the BMU with the data
                    self.codebook.matrix = self._update_codebook_voronoi(data[bootstrap_i],
                                                                         bmu,
                                                                         neighborhood)

                    # X2 is part of the Euclidean distance used during the finding of BMU
                    # for each line of data. As it is a fixed value, it can be ignored during
                    # the encounter of the BMU for each input data, but it is necessary for the calculation of the
                    # quantification error
                    fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(data[bootstrap_i], nan=0.0), np.nan_to_num(data[bootstrap_i], nan=0.0))
                    partial_qe = np.sqrt(abs(bmu[1] + fixed_euclidean_x2))
                    self.QE_expanded[bootstrap_i] = partial_qe
                    self.QE = np.mean(self.QE_expanded)

                    # Progress bar update
                    pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {round(self.QE,4)}")


                    # Update only the BMU of the vectors that participated in this training epoch
                    self._bmu[:,bootstrap_i] = bmu
                
                # Training without bootstrap
                else:
                    # Define the neighborhood for each specified radius
                    neighborhood = self.neighborhood.calculate(
                        self._distance_matrix, radius[i], self.codebook.nnodes)

                    # Find the BMU for the data
                    bmu = self._find_bmu(data, njb=njob, pace_size=self.pace_size)

                    # Update the BMU with the data
                    self.codebook.matrix = self._update_codebook_voronoi(data,
                                                                        bmu,
                                                                        neighborhood)

                    # X2 is part of the Euclidean distance used during the finding of BMU
                    # for each line of data. As it is a fixed value it can be ignored during
                    # the encounter of the BMU for each input data, but it is necessary for the calculation of the
                    # error quantification
                    fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(data, nan=0.0), np.nan_to_num(data, nan=0.0))
                    self.QE_expanded = np.sqrt(abs(bmu[1] + fixed_euclidean_x2))
                    self.QE = np.mean(self.QE_expanded)

                    # Progress bar update
                    pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {round(self.QE,4)}")
                    
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
                ## REVIEW THE QUANTIZATION ERROR CALCULATION
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
                        bmu = self._find_bmu(data[bootstrap_i], njb=njob, pace_size=self.pace_size)
                        
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

                        # Progress bar update
                        self.QE_expanded[bootstrap_i] = bmu[1]
                        self.QE = np.mean(self.QE_expanded)

                        pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {round(self.QE,4)}")

                    # Display imputed matrix in fine training
                    elif self.actual_train == "Fine":
                        if self.previous_epoch:
                            # fill in the missing values ​​in data with data from the previous interaction for BMU search
                            data[self.data_missing["indices"]] = self.data_missing["nan_values"]

                            # Find the BMU data
                            bmu = self._find_bmu(data[bootstrap_i], njb=njob, pace_size=self.pace_size)


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


                            # Progress bar update
                            fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(data[bootstrap_i], nan=0.0), np.nan_to_num(data[bootstrap_i], nan=0.0))
                            partial_qe = np.sqrt(abs(bmu[1] + fixed_euclidean_x2))
                            self.QE_expanded[bootstrap_i] = partial_qe
                            self.QE = np.mean(self.QE_expanded)
                            pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {round(self.QE,4)}. Reg:{round(reg,2)}")

                            
                            # Delete the data in the data variable
                            data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)

                        else:
                            # Find the BMU data
                            bmu = self._find_bmu(data[bootstrap_i], njb=njob, pace_size=self.pace_size)
                            
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

                            # Progress bar update
                            partial_qe = bmu[1]
                            self.QE_expanded[bootstrap_i] = partial_qe
                            self.QE = np.mean(self.QE_expanded)
                            pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {round(self.QE,4)}")

                    if self.history_plot:
                        if i%2 == 0:
                            self.plot_umatrix(figsize = (5,3),
                                         hits = True,
                                         save = True,
                                         file_name = f"{self.actual_train}_epoch{i+1}",
                                         bmu=bmu)

                    # Update only the BMU of the vectors that participated in this training epoch
                    self._bmu[:, bootstrap_i] = bmu
                # Not bootstrap training
                else:
                    # Define the neighborhood for each specified radius
                    neighborhood = self.neighborhood.calculate(
                        self._distance_matrix, radius[i], self.codebook.nnodes)

                    # Find the BMU and update the input data for the BMU search
                    # according to training

                    # Display incomplete matrix in rough training
                    if self.actual_train == "Rough":
                        # Find the BMU data
                        bmu = self._find_bmu(data, njb=njob, pace_size=self.pace_size)

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

                        # Progress bar update
                        self.QE_expanded = bmu[1]
                        self.QE = np.mean(self.QE_expanded)
                        pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {round(self.QE,4)}")

                    # Display matrix inputed in fine training
                    elif self.actual_train == "Fine":
                        if self.previous_epoch:

                            # Fill in the missing values ​​in data with data from the previous interaction for BMU search
                            data[self.data_missing["indices"]] = self.data_missing["nan_values"]

                            # Find the BMU data
                            bmu = self._find_bmu(data, njb=njob, pace_size=self.pace_size)

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


                            # Progress bar update
                            fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(data, nan=0.0), np.nan_to_num(data, nan=0.0))
                            self.QE_expanded = np.sqrt(abs(bmu[1] + fixed_euclidean_x2))
                            self.QE = np.mean(self.QE_expanded)
                            pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {round(self.QE,4)}. Reg:{round(reg,2)}")
                        else:
                            # Find the BMU data
                            bmu = self._find_bmu(data, njb=njob, pace_size=self.pace_size)

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

                            # Progress bar update
                            self.QE_expanded = bmu[1]
                            self.QE = np.mean(self.QE_expanded)
                            pbar.set_description(f"Epoch: {i+1}. Radius:{round(radius[i],2)}. QE: {round(self.QE,4)}")

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
                 project=False,
                 pace_size=None,
                 max_distance_memory_mb=None,
                 nan_distance_strategy="auto",
                 compute_dtype=None,
                 neuron_block_size=None):
        """
        Finds the BMU (Best Matching Unit) for each input vector.

        The calculation is performed in memory-bounded blocks. NumPy/BLAS is
        allowed to parallelize the matrix multiplication internally, avoiding
        nested parallelism between a Python thread pool and BLAS threads.

        Parameters
        ----------
        input_matrix : numpy.ndarray
            Input samples as rows and variables as columns.
        njb : int, default=-1
            Kept for backward compatibility. BMU search no longer creates a
            Python thread pool because the matrix multiplication is already
            parallelized by the numerical backend when available.
        nth : int, default=1
            Rank of the requested matching unit. ``1`` returns the BMU and
            ``2`` returns the second-best matching unit.
        project : bool, default=False
            Match projected data against the corresponding leading codebook
            variables.
        pace_size : int or None, default=100_000
            Maximum number of samples per block. The effective block size is
            also limited by ``max_distance_memory_mb``.
        max_distance_memory_mb : float, default=256
            Approximate upper memory limit for the temporary neuron-by-sample
            distance matrix.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(2, n_samples)``. Row 0 contains neuron indices
            and row 1 contains the corresponding distance values.
        """
        del njb  # Backward-compatible argument; BLAS handles parallelism.

        input_matrix = np.asarray(input_matrix)

        if input_matrix.ndim != 2:
            raise ValueError(
                "input_matrix must be a 2-D array with shape "
                "(n_samples, n_features)."
            )

        nth = int(nth)
        if nth < 1:
            raise ValueError("nth must be greater than or equal to 1.")

        if project:
            codebook = self.codebook.matrix[:, :input_matrix.shape[1]]
        else:
            codebook = self.codebook.matrix

        codebook = np.asarray(codebook)

        if input_matrix.shape[1] != codebook.shape[1]:
            raise ValueError(
                f"Feature mismatch: data has {input_matrix.shape[1]} columns, "
                f"but the codebook has {codebook.shape[1]}."
            )

        if nth > codebook.shape[0]:
            raise ValueError(
                f"nth={nth} exceeds the number of neurons "
                f"({codebook.shape[0]})."
            )

        missing = bool(np.isnan(input_matrix).any()) if project else self.missing

        return self._chunk_based_bmu_find(
            input_matrix=input_matrix,
            codebook=codebook,
            nth=nth,
            missing=missing,
            project=project,
            pace_size=pace_size,
            max_distance_memory_mb=max_distance_memory_mb,
            nan_distance_strategy=nan_distance_strategy,
            compute_dtype=compute_dtype,
            neuron_block_size=neuron_block_size,
        ).T

    def _find_bmu_top2(self,
                       input_matrix,
                       project=False,
                       pace_size=None,
                       max_distance_memory_mb=None,
                       nan_distance_strategy="auto",
                       compute_dtype=None,
                       neuron_block_size=None):
        """Return the first and second BMUs in a single distance pass."""
        input_matrix = np.asarray(input_matrix)

        if input_matrix.ndim != 2:
            raise ValueError(
                "input_matrix must be a 2-D array with shape "
                "(n_samples, n_features)."
            )

        if project:
            codebook = self.codebook.matrix[:, :input_matrix.shape[1]]
        else:
            codebook = self.codebook.matrix

        codebook = np.asarray(codebook)

        if input_matrix.shape[1] != codebook.shape[1]:
            raise ValueError(
                f"Feature mismatch: data has {input_matrix.shape[1]} columns, "
                f"but the codebook has {codebook.shape[1]}."
            )

        if codebook.shape[0] < 2:
            raise ValueError(
                "At least two neurons are required to calculate BMU1 and BMU2."
            )

        missing = bool(np.isnan(input_matrix).any()) if project else self.missing

        return self._chunk_based_bmu_find(
            input_matrix=input_matrix,
            codebook=codebook,
            nth=2,
            missing=missing,
            project=project,
            pace_size=pace_size,
            max_distance_memory_mb=max_distance_memory_mb,
            return_top_n=True,
            nan_distance_strategy=nan_distance_strategy,
            compute_dtype=compute_dtype,
            neuron_block_size=neuron_block_size,
        )


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
    

    def _nan_euclidean_distances_fast(self, codebook, data, strategy="auto"):
        """Compute sklearn-compatible nan-Euclidean distances efficiently.

        Parameters
        ----------
        codebook : ndarray, shape (n_nodes, n_features)
            Codebook vectors. This optimized implementation requires finite
            codebook values.
        data : ndarray, shape (n_samples, n_features)
            Input samples, possibly containing NaN values.
        strategy : {"auto", "vectorized", "grouped", "sklearn"}
            ``vectorized`` uses two BLAS products, ``grouped`` processes equal
            missing-value masks together, ``auto`` conservatively selects the
            vectorized implementation, and ``sklearn`` keeps the reference
            implementation. ``grouped`` is retained as an explicit experimental
            option rather than part of the default path.

        Returns
        -------
        ndarray, shape (n_nodes, n_samples)
            Euclidean distances with the same p/k regularization used by
            sklearn.metrics.pairwise.nan_euclidean_distances.
        """
        codebook = np.asarray(codebook)
        data = np.asarray(data)
        strategy = str(strategy).lower()

        if strategy not in {"auto", "vectorized", "grouped", "sklearn"}:
            raise ValueError(
                "nan distance strategy must be one of: auto, vectorized, "
                "grouped, sklearn."
            )
        if codebook.ndim != 2 or data.ndim != 2:
            raise ValueError("codebook and data must both be 2-D arrays.")
        if codebook.shape[1] != data.shape[1]:
            raise ValueError("codebook and data must have the same features.")
        if not np.isfinite(codebook).all():
            return nan_euclidean_distances(codebook, data)
        if strategy == "sklearn":
            return nan_euclidean_distances(codebook, data)

        observed = ~np.isnan(data)
        n_samples, n_features = data.shape
        n_observed = observed.sum(axis=1)

        if strategy == "auto":
            # Conservative default: the vectorized formulation was the most
            # consistently fast across repeated, random, and high-missingness
            # masks in the benchmark suite. The grouped implementation remains
            # available explicitly for experimentation, but is not selected
            # automatically because its Python-level grouping overhead can be
            # larger than the savings from reducing the matrix products.
            strategy = "vectorized"

        result_dtype = np.result_type(codebook.dtype, data.dtype, np.float32)
        W = np.ascontiguousarray(codebook, dtype=result_dtype)
        X = np.ascontiguousarray(np.nan_to_num(data, nan=0.0), dtype=result_dtype)
        distances_sq = np.empty((W.shape[0], n_samples), dtype=result_dtype)

        if strategy == "vectorized":
            mask_float = np.ascontiguousarray(observed, dtype=result_dtype)
            cross = W @ X.T
            distances_sq[:] = (W * W) @ mask_float.T
            distances_sq -= 2.0 * cross
            distances_sq += np.einsum("ij,ij->i", X, X, optimize=True)[None, :]
        else:
            unique_masks, inverse = np.unique(
                observed, axis=0, return_inverse=True
            )
            for mask_id, mask in enumerate(unique_masks):
                sample_idx = np.flatnonzero(inverse == mask_id)
                if not np.any(mask):
                    distances_sq[:, sample_idx] = np.nan
                    continue
                Wm = W[:, mask]
                Xm = X[sample_idx][:, mask]
                block = Wm @ Xm.T
                block *= -2.0
                block += np.einsum(
                    "ij,ij->i", Wm, Wm, optimize=True
                )[:, None]
                block += np.einsum(
                    "ij,ij->i", Xm, Xm, optimize=True
                )[None, :]
                distances_sq[:, sample_idx] = block

        np.maximum(distances_sq, 0.0, out=distances_sq)
        valid = n_observed > 0
        if np.any(valid):
            distances_sq[:, valid] *= (
                float(n_features) / n_observed[valid]
            )[None, :]
        distances_sq[:, ~valid] = np.nan
        np.sqrt(distances_sq, out=distances_sq)
        return distances_sq

    def _resolve_bmu_dtype(self, input_matrix, codebook, compute_dtype=None):
        """Resolve the numerical dtype used only by the BMU calculation."""
        value = getattr(self, "bmu_compute_dtype", "preserve") if compute_dtype is None else str(compute_dtype).lower()
        if value == "preserve":
            return np.result_type(input_matrix.dtype, codebook.dtype, np.float32)
        if value == "float32":
            return np.dtype(np.float32)
        if value == "float64":
            return np.dtype(np.float64)
        raise ValueError("compute_dtype must be 'preserve', 'float32', or 'float64'.")

    def _chunk_based_bmu_find(self,
                              input_matrix,
                              codebook,
                              y2=None,
                              nth=1,
                              missing=False,
                              project=False,
                              pace_size=None,
                              max_distance_memory_mb=None,
                              return_top_n=False,
                              nan_distance_strategy="auto",
                              compute_dtype=None,
                              neuron_block_size=None):
        """Find exact BMUs using memory-bounded sample and neuron blocks.

        Complete data use a two-axis exact search. Only a neuron-by-sample score
        tile is materialized, while the best candidates are merged across neuron
        tiles. Missing-aware searches retain the validated nan-Euclidean path.
        """
        input_matrix = np.asarray(input_matrix)
        codebook = np.asarray(codebook)
        if input_matrix.ndim != 2 or codebook.ndim != 2:
            raise ValueError("input_matrix and codebook must both be 2-D arrays.")
        if input_matrix.shape[1] != codebook.shape[1]:
            raise ValueError(
                f"Feature mismatch: data has {input_matrix.shape[1]} columns, "
                f"but the codebook has {codebook.shape[1]}."
            )

        dlen, nnodes = input_matrix.shape[0], codebook.shape[0]
        nth = int(nth)
        if nth < 1 or nth > nnodes:
            raise ValueError(f"nth must be between 1 and the number of neurons ({nnodes}).")
        if dlen == 0:
            if return_top_n:
                return (np.empty((nth, 0), dtype=np.intp),
                        np.empty((nth, 0), dtype=float))
            return np.empty((0, 2), dtype=float)

        dtype = self._resolve_bmu_dtype(input_matrix, codebook, compute_dtype)
        X = np.ascontiguousarray(input_matrix, dtype=dtype)
        W = np.ascontiguousarray(codebook, dtype=dtype)

        max_mb = getattr(self, "bmu_max_memory_mb", 256.0) if max_distance_memory_mb is None else float(max_distance_memory_mb)
        if max_mb <= 0:
            raise ValueError("max_distance_memory_mb must be greater than zero.")
        requested_sample_block = self.pace_size if pace_size is None else pace_size
        if requested_sample_block is None:
            requested_sample_block = dlen
        requested_sample_block = min(dlen, max(1, int(requested_sample_block)))

        cfg = getattr(self, "bmu_config", {
            "neuron_block_size": None,
            "memory_safety_factor": 2.0,
            "min_neuron_block_size": 1,
            "force_two_axis_blocking": False,
        })
        safety = float(cfg["memory_safety_factor"])
        usable_bytes = max(1, int(max_mb * 1024**2 / safety))
        itemsize = np.dtype(dtype).itemsize
        max_tile_elements = max(1, usable_bytes // itemsize)

        if missing:
            try:
                rough_training = self.actual_train == "Rough"
            except AttributeError:
                rough_training = False
            use_nan_euclidean = project or rough_training or not self.previous_epoch
        else:
            use_nan_euclidean = False

        # Missing data currently require all neurons in each validated kernel call.
        if use_nan_euclidean:
            memory_factor = 3
            sample_block = max(1, min(
                requested_sample_block,
                max_tile_elements // max(nnodes * memory_factor, 1),
            ))
            if return_top_n:
                all_indices = np.empty((nth, dlen), dtype=np.intp)
                all_distances = np.empty((nth, dlen), dtype=float)
            else:
                bmu = np.empty((dlen, 2), dtype=float)
            for low in range(0, dlen, sample_block):
                high = min(low + sample_block, dlen)
                distances = self._nan_euclidean_distances_fast(
                    W, X[low:high], strategy=nan_distance_strategy
                )
                columns = np.arange(high - low)
                if nth == 1:
                    idx = np.argmin(distances, axis=0)[None, :]
                    val = distances[idx[0], columns][None, :]
                else:
                    idx = np.argpartition(distances, kth=nth - 1, axis=0)[:nth]
                    val = np.take_along_axis(distances, idx, axis=0)
                    order = np.argsort(val, axis=0)
                    idx = np.take_along_axis(idx, order, axis=0)
                    val = np.take_along_axis(val, order, axis=0)
                if return_top_n:
                    all_indices[:, low:high] = idx
                    all_distances[:, low:high] = val
                else:
                    bmu[low:high, 0] = idx[nth - 1]
                    bmu[low:high, 1] = val[nth - 1]
            return (all_indices, all_distances) if return_top_n else bmu

        # Complete-data exact two-axis blocking.
        sample_block = requested_sample_block
        configured_neuron_block = (
            cfg["neuron_block_size"]
            if neuron_block_size is None else neuron_block_size
        )
        if configured_neuron_block is None:
            neuron_block = max(1, max_tile_elements // sample_block)
        else:
            neuron_block = max(1, int(configured_neuron_block))
        neuron_block = min(nnodes, neuron_block)

        # If the requested sample block is too large even for one neuron row,
        # shrink it. Otherwise preserve it and derive the neuron tile from memory.
        sample_block = min(sample_block, max_tile_elements)
        min_neuron_block = int(cfg["min_neuron_block_size"])
        if neuron_block < min_neuron_block and nnodes >= min_neuron_block:
            neuron_block = min_neuron_block
            sample_block = max(1, min(sample_block, max_tile_elements // neuron_block))
        if not cfg["force_two_axis_blocking"] and nnodes * sample_block <= max_tile_elements:
            neuron_block = nnodes

        W2 = np.einsum("ij,ij->i", W, W, optimize=True)
        if return_top_n:
            all_indices = np.empty((nth, dlen), dtype=np.intp)
            all_distances = np.empty((nth, dlen), dtype=float)
        else:
            bmu = np.empty((dlen, 2), dtype=float)

        for low in range(0, dlen, sample_block):
            high = min(low + sample_block, dlen)
            Xb = X[low:high]
            n_block = high - low
            best_scores = np.full((nth, n_block), np.inf, dtype=dtype)
            best_indices = np.full((nth, n_block), -1, dtype=np.intp)

            for nlow in range(0, nnodes, neuron_block):
                nhigh = min(nlow + neuron_block, nnodes)
                scores = W[nlow:nhigh] @ Xb.T
                scores *= -2
                scores += W2[nlow:nhigh, None]
                local_n = min(nth, nhigh - nlow)
                if local_n == 1:
                    local_idx = np.argmin(scores, axis=0)[None, :]
                else:
                    local_idx = np.argpartition(scores, kth=local_n - 1, axis=0)[:local_n]
                local_scores = np.take_along_axis(scores, local_idx, axis=0)
                local_idx = local_idx + nlow

                candidate_scores = np.concatenate((best_scores, local_scores), axis=0)
                candidate_indices = np.concatenate((best_indices, local_idx), axis=0)
                keep = np.argpartition(candidate_scores, kth=nth - 1, axis=0)[:nth]
                best_scores = np.take_along_axis(candidate_scores, keep, axis=0)
                best_indices = np.take_along_axis(candidate_indices, keep, axis=0)

            order = np.argsort(best_scores, axis=0)
            best_scores = np.take_along_axis(best_scores, order, axis=0)
            best_indices = np.take_along_axis(best_indices, order, axis=0)
            if return_top_n:
                all_indices[:, low:high] = best_indices
                all_distances[:, low:high] = best_scores
            else:
                bmu[low:high, 0] = best_indices[nth - 1]
                bmu[low:high, 1] = best_scores[nth - 1]

        return (all_indices, all_distances) if return_top_n else bmu

    @property
    def calculate_quantization_error(self):
        return self.QE
    
    @property
    def calculate_quantization_error_expanded(self):
        return self.QE_expanded

    
    @property
    def topographic_error(self):
        """
        Calculate the topographic error of the trained SOM.

        The topographic error measures the proportion of samples whose first
        and second Best Matching Units (BMU1 and BMU2) are not adjacent on the
        SOM lattice.

        Adjacency depends on the lattice geometry.

        Hexagonal lattice
        -----------------
        A neuron has six immediate neighbors. In the grid-distance matrix,
        immediate hexagonal neighbors have distance equal to 1.

        Rectangular lattice
        -------------------
        The rectangular lattice uses an 8-connected neighborhood:

            NW   N   NE
            \\ | /
            W -- X -- E
            / | \\
            SW   S   SE

        The rectangular grid stores squared Euclidean distances:

            orthogonal neighbor -> 1
            diagonal neighbor   -> 2

        Therefore, BMU1 and BMU2 are considered adjacent when their squared
        grid distance is less than or equal to 2.

        Both planar and toroidal topology are handled automatically through
        ``self._distance_matrix``.

        Returns
        -------
        float
            Fraction of input samples for which BMU1 and BMU2 are not
            topological neighbors. The result ranges from 0 to 1.
        """

        top2_indices, _ = self._find_bmu_top2(
            self.get_data,
            pace_size=self.pace_size,
        )

        bmus1 = top2_indices[0].astype(int)
        bmus2 = top2_indices[1].astype(int)

        if self.lattice == "rect":

            cols, rows = self.mapsize

            row1, col1 = np.divmod(bmus1, cols)
            row2, col2 = np.divmod(bmus2, cols)

            row_delta = np.abs(row1 - row2)
            col_delta = np.abs(col1 - col2)

            if self.mapshape == "toroid":
                row_delta = np.minimum(row_delta, rows - row_delta)
                col_delta = np.minimum(col_delta, cols - col_delta)

            adjacent = (
                (row_delta <= 1)
                & (col_delta <= 1)
                & ((row_delta + col_delta) > 0)
            )

        elif self.lattice == "hexa":

            distances = self._distance_matrix[
                bmus1,
                bmus2,
            ]

            adjacent = np.isclose(
                distances,
                1.0,
            )

        else:

            raise ValueError(
                "Unsupported lattice. "
                "Expected 'rect' or 'hexa', "
                f"received {self.lattice!r}."
            )

        return float(
            np.mean(
                ~adjacent
            )
        )
        

    def build_umatrix(self, expanded=False, log=False):
        """
        Calculate the U-Matrix from trained neuron weights.

        Map convention
        --------------
        mapsize = (columns, rows)
        NumPy shape = (rows, columns)

        Expanded neighbor order
        -----------------------
        Hexagonal (6):
            0 = right
            1 = down-right
            2 = down-left
            3 = left
            4 = up-left
            5 = up-right

            The horizontal offset of diagonal neighbors follows the odd-r row
            parity used by IntraSOM.

        Rectangular (8):
            0 = right
            1 = down-right
            2 = down
            3 = down-left
            4 = left
            5 = up-left
            6 = up
            7 = up-right

            Diagonal distances are intentionally included so the rectangular
            U-Matrix has a complete expanded representation between neurons.

        Parameters
        ----------
        expanded : bool, default=False
            If True, return every neighbor distance with shape
            (rows, cols, n_neighbors). If False, return the mean neighbor
            distance for each neuron with shape (rows, cols).
        log : bool, default=False
            Apply the natural logarithm, preserving the historical IntraSOM
            behavior.
        """

        cols, rows = self.mapsize

        weights = np.asarray(
            self.codebook.matrix,
            dtype=float,
        ).reshape(
            rows,
            cols,
            self.codebook.matrix.shape[1],
        )

        row_grid, col_grid = np.indices(
            (rows, cols),
            dtype=int,
        )

        if self.lattice == "hexa":
            # Historical odd-r convention used by IntraSOM.
            # Even rows use ii[1] in the original implementation;
            # odd rows use ii[0].
            dx_even = np.array(
                [1, 0, -1, -1, -1, 0],
                dtype=int,
            )
            dx_odd = np.array(
                [1, 1, 0, -1, 0, 1],
                dtype=int,
            )
            dy = np.array(
                [0, 1, 1, 0, -1, -1],
                dtype=int,
            )

            n_neighbors = 6
            even_rows = (row_grid % 2) == 0

        elif self.lattice == "rect":
            # Clockwise order starting at the right neighbor.
            offsets = np.array(
                [
                    [1, 0],    # right
                    [1, 1],    # down-right
                    [0, 1],    # down
                    [-1, 1],   # down-left
                    [-1, 0],   # left
                    [-1, -1],  # up-left
                    [0, -1],   # up
                    [1, -1],   # up-right
                ],
                dtype=int,
            )

            n_neighbors = 8

        else:
            raise ValueError(
                "lattice must be 'hexa' or 'rect'. "
                f"Received: {self.lattice!r}"
            )

        um = np.full(
            (rows, cols, n_neighbors),
            np.nan,
            dtype=float,
        )

        for k in range(n_neighbors):

            if self.lattice == "hexa":
                dx = np.where(
                    even_rows,
                    dx_even[k],
                    dx_odd[k],
                )
                step_y = dy[k]
            else:
                dx = offsets[k, 0]
                step_y = offsets[k, 1]

            neighbor_rows = row_grid + step_y
            neighbor_cols = col_grid + dx

            if self.mapshape == "toroid":
                neighbor_rows %= rows
                neighbor_cols %= cols

                neighbor_weights = weights[
                    neighbor_rows,
                    neighbor_cols,
                ]

                um[:, :, k] = np.linalg.norm(
                    weights - neighbor_weights,
                    axis=2,
                )

            elif self.mapshape == "planar":
                valid = (
                    (neighbor_rows >= 0)
                    & (neighbor_rows < rows)
                    & (neighbor_cols >= 0)
                    & (neighbor_cols < cols)
                )

                if np.any(valid):
                    source_weights = weights[valid]
                    neighbor_weights = weights[
                        neighbor_rows[valid],
                        neighbor_cols[valid],
                    ]

                    um[:, :, k][valid] = np.linalg.norm(
                        source_weights - neighbor_weights,
                        axis=1,
                    )

            else:
                raise ValueError(
                    "mapshape must be 'planar' or 'toroid'. "
                    f"Received: {self.mapshape!r}"
                )

        if expanded:
            return (
                np.log1p(um)
                if log
                else um
            )

        umat = np.nanmean(
            um,
            axis=2,
        )

        return (
            np.log1p(umat)
            if log
            else umat
        )

        return np.log(reduced) if log else reduced

        
    
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
                                     facecolor= mpl.colormaps["jet"](norm(umat[j][i])),
                                     alpha=1)#, edgecolor='black')

                ax.add_patch(hex)

                # Right Hexagon
                if not np.isnan(um[j, i, 0]):
                    hex = RegularPolygon((xx[(j, i)]*2+1,
                                          yy[(j,i)]*2),
                                         numVertices=6,
                                         radius=1/np.sqrt(3),
                                         facecolor=mpl.colormaps["jet"](norm(um[j,i,0])),
                                         alpha=1)
                    ax.add_patch(hex)

                # Upper Right Hexagon
                if not np.isnan(um[j, i, 1]):
                    hex = RegularPolygon((xx[(j, i)]*2+0.5,
                                          yy[(j,i)]*2+(np.sqrt(3)/2)),
                                         numVertices=6,
                                         radius=1/np.sqrt(3),
                                         facecolor=mpl.colormaps["jet"](norm(um[j,i,1])),
                                         alpha=1)
                    ax.add_patch(hex)

                # Upper Left Hexagon
                if not np.isnan(um[j, i, 2]):
                    hex = RegularPolygon((xx[(j, i)]*2-0.5,
                                          yy[(j,i)]*2+(np.sqrt(3)/2)),
                                         numVertices=6,
                                         radius=1/np.sqrt(3),
                                         facecolor=mpl.colormaps["jet"](norm(um[j,i,2])),
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
        
        sorted_dict = {k+1: rep_samples_dic[k] for k in sorted(rep_samples_dic)}

        return sorted_dict


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
