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
        Initialization of the map by using just the first two eigenvalues and
        eigenvectors. The initialization is done in the following steps:
        1. Transformation of input data and creation of a template matrix
        2. Creation of a matrix with scaling factors for each principal component (PC)
        3. Obtaining and normalization of the eigenvector(s)
        4. Linear combination for each node of the template matrix with the PC

        Args:
            data: data to use for the initialization

        Returns:
            Initialized matrix with same dimension as input data.
        """
        cols = self.mapsize[1]
        coord = None
        pca_components = None

        me = np.mean(data, 0)
        data = (data - me)
        tmp_matrix = np.tile(me, (self.nnodes, 1))

        if np.min(self.mapsize) > 1:
            coord = np.zeros((self.nnodes, 2))
            pca_components = 2

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i / cols)  # x
                coord[i, 1] = int(i % cols)  # y

        elif np.min(self.mapsize) == 1:
            coord = np.zeros((self.nnodes, 1))
            pca_components = 1

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i % cols)  # y

        mx = np.max(coord, axis=0)
        mn = np.min(coord, axis=0)
        coord = (coord - mn)/(mx-mn)
        coord = (coord - .5)*2

        pca = PCA(n_components=pca_components, svd_solver='randomized')

        pca.fit(data)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
        eigvec = ((eigvec.T/norms)*eigval).T

        for j in range(self.nnodes):
            for i in range(eigvec.shape[0]):
                tmp_matrix[j, :] = tmp_matrix[j, :] + coord[j, i]*eigvec[i, :]

        self.matrix = np.around(tmp_matrix, decimals=6)
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
        Finds the Manhattan distance matrix (L1) of a neural network node
        for all others, for a rectangular lattice in a map with
        planar topology.

        Args:
            node_ind: neural network node index, between 0 and nnodes-1.

        Returns:
            Returns array of distances from this node to all other nodes in the
            grid, on a planar map.

        """
        # Separate column and row values
        rows, cols = self.mapsize

        # Generate the xy coordinates of the BMUs for a rectangular grid
        coordinates = self.generate_rec_lattice(rows, cols)

        # Find the Manhattan distances for a rectangular grid through
        # its coordinates
        dist = np.array(abs(coordinates[ind] - coordinates[node_ind]).sum() \
            for ind in range(len(coordinates)))

        return dist

    def _rect_dist_tor(self, node_ind):
        """
        Finds the matrix of distances from a neural network node to all
        others, for a hexagonal lattice in a map with toroidal topology.
        Args:
            node_ind: Neural network node index, between 0 and nnodes-1.

        Returns:
            Returns the distances from this node to all other grid nodes, in a
            toroidal map.

        """
        rows, cols = self.mapsize

        # Generate the xy coordinates of the BMUs for a rectangular grid
        coordinates = self.generate_hex_lattice(rows, cols)

        # Extends the distance search to the neighborhood created by the toroidal topology, creating periodicity of the data
        toroid_neigh = [[0, 0], [cols, 0], [cols, rows], [0, -rows],
            [-cols, 0], [0, -rows], [-cols, -rows]]

        # Calculate the distances in the toroidal topology, finding all possible distances according to toroid_neigh and
        # selecting the smallest
        dist = np.array(
            [min([abs((coordinates[ind] + [neig]) - coordinates[node_ind]).sum()\
             for neig in toroid_neigh]) for ind in range(len(coordinates))])

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
        Finds the Manhattan distance matrix (L1) of a neural network node
        for all others, for a rectangular lattice in a map with
        toroidal topology.

        Args:
            node_ind: Neural network node index, between 0 and nnodes-1.

        Returns:
            Returns an array of distances from this node to all other nodes in the
            grid, in a toroidal map.
        """

        # Separate column and row values
        cols, rows = self.mapsize

        # Generate the BMUs xyz coordinates for a hexagonal grid
        coordinates = self.generate_oddr_cube_lattice(cols, rows)
        
        
        # Extend the distance search to the neighborhood created by the toroidal
        # topology, creating data periodicity
        toroid_neigh = self.toroid_neighborhood(cols, rows)

        # Calculate the distances in the toroidal topology, finding all
        # possible distances according to toroid_neigh and selecting the smallest
        dist = np.zeros(((cols*rows),9))
        for i in range(cols*rows):
            if i >= node_ind:
                for j, neig in enumerate(toroid_neigh):
                    dist[i,j] = self.cube_distance(coordinates[i]+neig, coordinates[node_ind], dist_factor=self.dist_factor)


        return np.min(dist, axis=1).astype(int)

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
