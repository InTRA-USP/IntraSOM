import numpy as np
from sklearn.decomposition import PCA

class Codebook(object):
    """
    Class for creating the SOM codebook. The codebook is the matrix of weights that is
    trained in its competitive, collaborative and adaptive synaptic processes.
    """

    def __init__(self, mapsize, lattice, mapshape, dist_factor):
        self.lattice = lattice
        self.mapshape = mapshape
        self.dist_factor = dist_factor

        if 2 == len(mapsize):
            _size = [1, np.max(mapsize)] if 1 == np.min(mapsize) else mapsize

        elif 1 == len(mapsize):
            _size = [1, mapsize[0]]
            print('Sizemap input was considered\
                as the number of neural network nodes')
            print('The size of the map \
                is [{dlen},{dlen}]'.format(dlen=int(mapsize[0] / 2)))
        else:
            pass

        self.mapsize = _size
        self.nnodes = self.mapsize[0] * self.mapsize[1]
        self.matrix = np.asarray(self.mapsize)
        self.initialized = False


    @property
    def get_matrix(self):
        """
        Returns the array stored in the current object.

        Returns:

        matrix: ndarray
        The array stored in the current object.
        """
        return self.matrix

    def random_initialization(self, data):
        """
        Random type initialization
        Args:
            data: data used for initialization

        Returns:
            Array initialized with the same dimensions as the input data.
        """

        # Constructs an array by repeating the smallest and largest data values ​​with
        # extension being the number of nodes required
        # Modification for nanmin and nanmax to take into account the
        # existence of NaN data
        mn = np.tile(np.nanmin(data, axis=0), (self.nnodes, 1))
        mx = np.tile(np.nanmax(data, axis=0), (self.nnodes, 1))

        # Minimum value + (maximum value - minimum value)*(array of random
        # numbers [0,1] with the format of number of nodes of lines and number of
        # columns of data in columns. A training matrix randomly
        # initialized.
        #np.random.seed(0)
        self.matrix = mn + (mx - mn) *\
            (np.random.rand(self.nnodes, data.shape[1]))
        self.initialized = True

    def pca_linear_initialization(self, data):
        """
        Initializes the SOM codebook using the first principal components
        of the input data.

        The SOM map convention is:

            mapsize = (columns, rows)

        For a two-dimensional map, neuron coordinates are generated using
        row-major indexing:

            row = index // columns
            col = index % columns

        The first spatial coordinate corresponds to the column (x) and the
        second to the row (y).

        The initialization is performed in the following steps:

            1. Center the input data around its mean.
            2. Generate normalized spatial coordinates for each SOM neuron.
            3. Compute the first principal component(s) of the input data.
            4. Scale the principal directions by their explained variance.
            5. Distribute the neuron vectors along those directions.

        Parameters
        ----------
        data : numpy.ndarray
            Input data used to initialize the SOM codebook.
            Rows represent samples and columns represent variables.

        Returns
        -------
        None
            The initialized codebook is stored in ``self.matrix``.
        """

        # --------------------------------------------------
        # INPUT DATA
        # --------------------------------------------------

        data = np.asarray(
            data,
            dtype=float
        )

        # --------------------------------------------------
        # MAP GEOMETRY
        #
        # IntraSOM convention:
        # mapsize = (columns, rows)
        # --------------------------------------------------

        cols, rows = self.mapsize

        # --------------------------------------------------
        # CENTER DATA
        # --------------------------------------------------

        me = np.mean(
            data,
            axis=0
        )

        centered_data = (
            data - me
        )

        # Start every neuron at the mean vector.
        tmp_matrix = np.tile(
            me,
            (
                self.nnodes,
                1
            )
        )

        # --------------------------------------------------
        # SOM SPATIAL COORDINATES
        # --------------------------------------------------

        if cols > 1 and rows > 1:

            # Two-dimensional SOM
            pca_components = 2

            coord = np.zeros(
                (
                    self.nnodes,
                    2
                ),
                dtype=float
            )

            for node_ind in range(
                self.nnodes
            ):

                # Row-major indexing:
                #
                # 0  1  2  ... cols-1
                # cols ...
                #
                row = node_ind // cols
                col = node_ind % cols

                # Spatial coordinates:
                #
                # axis 0 -> x -> column
                # axis 1 -> y -> row
                coord[
                    node_ind,
                    0
                ] = col

                coord[
                    node_ind,
                    1
                ] = row

        else:

            # One-dimensional SOM:
            #
            # mapsize may be:
            #     (N, 1)
            # or
            #     (1, N)
            #
            # In both cases there is only one spatial dimension.

            pca_components = 1

            coord = np.arange(
                self.nnodes,
                dtype=float
            ).reshape(
                -1,
                1
            )

        # --------------------------------------------------
        # NORMALIZE SPATIAL COORDINATES TO [-1, 1]
        # --------------------------------------------------

        coord_max = np.max(
            coord,
            axis=0
        )

        coord_min = np.min(
            coord,
            axis=0
        )

        coord_range = (
            coord_max - coord_min
        )

        # Protection against division by zero.
        #
        # Normally this only matters for a map containing
        # a single neuron.
        coord_range[
            coord_range == 0
        ] = 1.0

        coord = (
            coord - coord_min
        ) / coord_range

        coord = (
            coord - 0.5
        ) * 2.0

        # --------------------------------------------------
        # PCA
        # --------------------------------------------------

        pca = PCA(
            n_components=pca_components,
            svd_solver="randomized"
        )

        pca.fit(
            centered_data
        )

        eigvec = pca.components_

        eigval = (
            pca.explained_variance_
        )

        # --------------------------------------------------
        # NORMALIZE PCA DIRECTIONS
        # --------------------------------------------------

        norms = np.sqrt(
            np.einsum(
                "ij,ij->i",
                eigvec,
                eigvec
            )
        )

        # Protection against division by zero.
        norms[
            norms == 0
        ] = 1.0

        eigvec = (
            (
                eigvec.T / norms
            )
            * eigval
        ).T

        # --------------------------------------------------
        # INITIALIZE EACH NEURON
        #
        # mean
        #   +
        # spatial coordinate along PC1
        #   +
        # spatial coordinate along PC2
        # --------------------------------------------------

        for node_ind in range(
            self.nnodes
        ):

            for component_ind in range(
                eigvec.shape[0]
            ):

                tmp_matrix[
                    node_ind,
                    :
                ] += (
                    coord[
                        node_ind,
                        component_ind
                    ]
                    *
                    eigvec[
                        component_ind,
                        :
                    ]
                )

        # --------------------------------------------------
        # SAVE INITIALIZED CODEBOOK
        # --------------------------------------------------

        self.matrix = np.around(
            tmp_matrix,
            decimals=6
        )

        self.initialized = True
    
    def pretrain(self):
        self.initialized = True

    def pretrain(self):
        self.initialized = True


    def grid_dist(self, node_ind):
        """
        Calculates distances on the grid for maps with planar or toroidal
        topology and with rectangular or hexagonal lattice.

        Args:
            node_ind: neural network node index, between 0 and nnodes-1.

        Returns:
            Returns the distances from this node to all other grid nodes,
            within the parameters specified in the SOM object.

        """

        # Define which function to call for each lattice and topology
        if self.mapshape == 'planar':
            if self.lattice == 'rect':
                return self._rect_dist_plan(node_ind)

            elif self.lattice == 'hexa':
                return self._hexa_dist_plan(node_ind)

        if self.mapshape == 'toroid':
            if self.lattice == 'rect':
                return self._rect_dist_tor(node_ind)

            elif self.lattice == 'hexa':
                return self._hexa_dist_tor(node_ind)

    def _rect_dist_plan(self, node_ind):
        """
        Calculates Manhattan distances from one node to all other nodes
        in a rectangular lattice with planar topology.

        Parameters
        ----------
        node_ind : int
            Node index between 0 and nnodes - 1.

        Returns
        -------
        numpy.ndarray
            Manhattan distance from node_ind to every node.
        """

        # mapsize = (columns, rows)
        cols, rows = self.mapsize

        # generate_rec_lattice expects (n_rows, n_columns)
        coordinates = self.generate_rec_lattice(
            rows,
            cols
        )

        dist = np.array([
            np.abs(
                coordinates[ind]
                - coordinates[node_ind]
            ).sum()
            for ind in range(len(coordinates))
        ])

        return dist

    def _rect_dist_tor(self, node_ind):
        """
        Calculates Manhattan distances from one node to all other nodes
        in a rectangular lattice with toroidal topology.

        Parameters
        ----------
        node_ind : int
            Node index between 0 and nnodes - 1.

        Returns
        -------
        numpy.ndarray
            Distance from node_ind to every node in the toroidal
            rectangular grid.
        """

        cols, rows = self.mapsize

        # Convert linear index to row/column coordinates
        node_row = node_ind // cols
        node_col = node_ind % cols

        dist = np.zeros(self.nnodes, dtype=int)

        for ind in range(self.nnodes):

            row = ind // cols
            col = ind % cols

            # Direct distances
            dx = abs(col - node_col)
            dy = abs(row - node_row)

            # Toroidal shortest distances
            dx = min(dx, cols - dx)
            dy = min(dy, rows - dy)

            # Manhattan distance
            dist[ind] = dx + dy

        return dist

    def _hexa_dist_plan(self, node_ind):
        """

        Finds the matrix of distances from a neural network node to all
        others, for a hexagonal lattice in a map with planar topology.

        Args:
            node_ind: neural network node index, between 0 and nnodes-1.

        Returns:
            Returns the distances from this node to all other grid nodes, in a
            planar map.

        """
        cols, rows = self.mapsize

        # Generate x,y coordinates for a hexagonal grid
        coordinates = self.generate_oddr_cube_lattice(cols, rows)

        # Find the manhatan distances for a hexagonal grid via their xy coordinates
        dist = np.array([self.cube_distance(coordinates[node_ind], coordinates[i], dist_factor=self.dist_factor)\
         for i in range(len(coordinates))])

        return dist.astype(int)

    def _hexa_dist_tor(self, node_ind):
        """
        Calculates grid distances from one node to all other nodes
        in a hexagonal lattice with toroidal topology.

        Parameters
        ----------
        node_ind : int
            Node index between 0 and nnodes - 1.

        Returns
        -------
        numpy.ndarray
            Minimum toroidal grid distance from node_ind
            to every node.
        """

        # mapsize = (columns, rows)
        cols, rows = self.mapsize

        # Validate node index
        if not 0 <= node_ind < self.nnodes:
            raise IndexError(
                f"node_ind must be between 0 and "
                f"{self.nnodes - 1}. "
                f"Received: {node_ind}."
            )

        # Cubic coordinates for the odd-r hexagonal lattice
        coordinates = self.generate_oddr_cube_lattice(
            cols,
            rows
        )

        # Periodic copies required by toroidal topology
        toroid_neigh = self.toroid_neighborhood(
            cols,
            rows
        )

        # One row per neuron and one column per
        # periodic representation
        dist = np.zeros(
            (
                self.nnodes,
                len(toroid_neigh)
            ),
            dtype=float
        )

        # Calculate distances for ALL neurons
        for i in range(self.nnodes):

            for j, neig in enumerate(toroid_neigh):

                dist[i, j] = self.cube_distance(
                    coordinates[i] + neig,
                    coordinates[node_ind],
                    dist_factor=self.dist_factor
                )

        # Keep shortest toroidal path
        return np.min(
            dist,
            axis=1
        ).astype(int)

    def toroid_neighborhood(self, cols, rows):
        """
        Function to generate the cubic coordinates in the toroidal neighborhood for a given
        map size, in order: [Center, Right, Bottom Right, Bottom,
        Bottom Left, Left, Top Left, Top, Top Right].

        Args:

            cols: number of map columns to be generated.

            rows: number of rows of the map to be generated.

        Returns:
            List of distance coordinates
        """
        toroid_neigh = [[0, 0],
                        [cols, 0],
                        [cols, rows],
                        [0, rows],
                        [-cols, rows],
                        [-cols, 0],
                        [-cols, -rows],
                        [0,-rows],
                        [cols, -rows]]

        return [self.oddr_to_cube(i[0], i[1]) for i in toroid_neigh]


    def generate_oddr_cube_lattice(self, n_columns, n_rows):
        """
        Function to generate cubic coordinates in [x,y,z] format for an odd-r
        hexagonal grid (odd lines shifted to the right) for a
        predetermined number of columns and row.
        Args:
            n_columns: number of columns

            n_rows: number of rows

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
                x_coord.append(x)
                y_coord.append(y)
                z_coord.append(z)

        coordinates = np.column_stack([x_coord, y_coord, z_coord])
        return coordinates

    def cube_distance(self, a, b, dist_factor = 2):
        """
        Calculates the Euclidean distance between two cubic coordinates
        Args:
            a: First cubic coordinate [x,y,z]
            b: Second cubic coordinate [x,y,z]

        Returns:

            Manhattan distance between coordinates.
        """
        return ((abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]))/2)**dist_factor

    def oddr_to_cube(self, col, row):
        """
        Transforms rectangular coordinates to cubic.

        Args:
            coord: Coordinate you want to transform.

        Returns:
            Cubic coordinate in [x,y,z] format
        """

        x = col - (row - (row & 1)) / 2
        z = row
        y = -x-z
        return [x, y, z]


    def generate_rec_lattice(self, n_rows, n_columns):
        """
        Generates the xy coordinates of the BMUs for a rectangular grid.

        Args:
            n_rows: Number of rows in the Kohonen map.
            n_columns: Number of columns in the Kohonen map.

        returns:
            Coordinates in the [x,y] format for the BMUs in a rectangular grid.
        """
        x_coord = []
        y_coord = []
        for j in range(n_rows):
            for i in range(n_columns):
                x_coord.append(i)
                y_coord.append(j)
        coordinates = np.column_stack([x_coord, y_coord])
        return coordinates

    def generate_hex_lattice(self, n_columns, n_rows):
        """
        Generates the xy coordinates of the BMUs for an odd-r hexagonal grid.
        Args:
            n_rows: Number of rows in the Kohonen map.
            n_columns: Number of columns in the Kohonen map.

        Returns:
            Coordinates in the [x,y] format for the bmus in a hexagonal grid.
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
