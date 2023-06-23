import numpy as np

class Codebook(object):
    """
    Classe para criaçação do codebook SOM. O codebook é a matriz de pesos que é
    treinada nos seus processos competitivos, colaborativos e de adaptação
    sináptica.
    """

    def __init__(self, mapsize, lattice, mapshape):
        self.lattice = lattice
        self.mapshape = mapshape

        if 2 == len(mapsize):
            _size = [1, np.max(mapsize)] if 1 == np.min(mapsize) else mapsize

        elif 1 == len(mapsize):
            _size = [1, mapsize[0]]
            print('Input de sizemap foi considerado\
                como o número de nós da rede neural')
            print('O tamanho do mapa \
                é [{dlen},{dlen}]'.format(dlen=int(mapsize[0] / 2)))
        else:
            pass

        self.mapsize = _size
        self.nnodes = self.mapsize[0] * self.mapsize[1]
        self.matrix = np.asarray(self.mapsize)
        self.initialized = False

    @property
    def get_matrix(self):
        """
        Retorna a matriz armazenada no objeto atual.

        Retorno:

        matrix: ndarray
        A matriz armazenada no objeto atual.
        """
        return self.matrix

    def random_initialization(self, data):
        """
        Inicialização do tipo randômica
        Args:
            data: dados utilizados para a inicialização

        Returns:
            Matriz inicializada com as mesmas dimensões dos dados de entrada.
        """

        # Controi um array repetindo o menor e o maior valor dos dados com
        # extensão sendo o número de nós necessário
        # Modificação para nanmin e nanmax para que se leve em consideração a
        # existência de dados NaN
        mn = np.tile(np.nanmin(data, axis=0), (self.nnodes, 1))
        mx = np.tile(np.nanmax(data, axis=0), (self.nnodes, 1))

        # Valor mínimo + (valor máximo - valor mínimo)*(matriz de números
        # aleatórios [0,1] com o formato de numero de nós de linhas e número de
        # colunas dos dados nas colunas. Uma matriz de treinamento inicializada
        # randomicamente.
        #np.random.seed(0)
        self.matrix = mn + (mx - mn) *\
            (np.random.rand(self.nnodes, data.shape[1]))
        self.initialized = True
    
    def pretrain(self):
        self.initialized = True


    def grid_dist(self, node_ind):
        """
        Calcula as distâncias no grid para mapas com topologia planar ou
        toroidal e com lattice retangular ou hexagonal.

        Args:
            node_ind: índice do nó da rede neural, entre 0 e nnodes-1.

        Retorna:
            Retorna as distâncias desse nó para todos os outros nós do grid,
            dentro dos parâmetros especificados no objeto SOM.

        """

        # Definir qual função chamar para cada lattice e topologia
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
        Encontra a matriz de distâncias Manhattan (L1) de um nó da rede neural
        para todos os outros, para um lattice retangular em um mapa com
        topologia planar.

        Args:
            node_ind: índice do nó da rede neural, entre 0 e nnodes-1.

        Returns:
            Retorna array de distâncias desse nó para todos os outros nós do
            grid, num mapa planar.

        """
        # Separar os valores de colunas e linhas
        rows, cols = self.mapsize

        # Gera as coordenadas xy dos bmus para um grid retangular
        coordinates = self.generate_rec_lattice(rows, cols)

        # Encontra as distâncias manhatan para um grid retangular através de
        # suas coordenadas
        dist = np.array(abs(coordinates[ind] - coordinates[node_ind]).sum() \
            for ind in range(len(coordinates)))

        return dist

    def _rect_dist_tor(self, node_ind):
        """
        Encontra a matriz de distâncias de um nó da rede neural para todos os
        outros, para um lattice hexagonal em um mapa com topologia toroidal.
        Args:
            node_ind: Índice do nó da rede neural, entre 0 e nnodes-1.

        Returns:
            Retorna as distâncias desse nó para todos os outros nós do grid, num
            mapa toroidal.

        """
        rows, cols = self.mapsize

        # Gera as coordenadas xy dos bmus para um grid retangular
        coordinates = self.generate_hex_lattice(rows, cols)

        # Amplia a busca de distâncias para a vizinhança criada pela topologia toroidal, criando periodicidade dos dados
        toroid_neigh = [[0, 0], [cols, 0], [cols, rows], [0, -rows],
            [-cols, 0], [0, -rows], [-cols, -rows]]

        # Calcula as distâncias na topologia toroidal, encontrando todas as distâncias possíveis segundo torid_neigh e
        # selecionando o menor
        dist = np.array(
            [min([abs((coordinates[ind] + [neig]) - coordinates[node_ind]).sum()\
             for neig in toroid_neigh]) for ind in range(len(coordinates))])

        return dist

    def _hexa_dist_plan(self, node_ind):
        """

        Encontra a matriz de distâncias de um nó da rede neural para todos os
        outros, para um lattice hexagonal em um mapa com topologia planar.

        Args:
            node_ind: índice do nó da rede neural, entre 0 e nnodes-1.

        Returns:
            Retorna as distâncias desse nó para todos os outros nós do grid, num
            mapa planar.

        """
        cols, rows = self.mapsize

        # Gera coordenadas x,y para um grid hexagonal
        coordinates = self.generate_oddr_cube_lattice(cols, rows)

        # Encontra as distâncias manhatan para um grid hexagonal através de suas coordenadas xy
        dist = np.array([self.cube_distance(coordinates[node_ind], coordinates[i])**2\
         for i in range(len(coordinates))])

        return dist

    def _hexa_dist_tor(self, node_ind):
        """

        Encontra a matriz de distâncias Manhattan (L1) de um nó da rede neural
        para todos os outros, para um lattice retangular em um mapa com
        topologia toroidal.

        Args:
            node_ind: Índice do nó da rede neural, entre 0 e nnodes-1.

        Retorna:
            Retorna um array de distâncias desse nó para todos os outros nós do
            grid, num mapa toroidal.
        """

        # Separar os valores de colunas e linhas
        cols, rows = self.mapsize

        # Gera as coordenadas xyz dos bmus para um grid hexagonal 
        coordinates = self.generate_oddr_cube_lattice(cols, rows)
        
        
        # Amplia a busca de distâncias para a vizinhança criada pela topologia
        # toroidal, criando periodicidade dos dados
        toroid_neigh = self.toroid_neighborhood(cols, rows)

        # Calcula as distâncias na topologia toroidal, encontrando todas as
        # distâncias possíveis segundo torid_neigh e selecionando o menor
        dist = np.zeros(((cols*rows),9))
        for i in range(cols*rows):
            if i >= node_ind:
                for j, neig in enumerate(toroid_neigh):
                    dist[i,j] = self.cube_distance(coordinates[i]+neig, coordinates[node_ind])


        return np.min(dist, axis=1).astype(int)

    def toroid_neighborhood(self, cols, rows):
        """
        Função para gerar as coordenadas cúbicas na vizinhança toroidal para um dado
        tamanho de mapa, na ordem: [Central, Direita, Direita Inferior, Inferior,
        Esquerda Inferior, Esquerda, Esquerda Superior, Superior, Direita Superior].

        Args:

            cols: quantidade de colunas do mapa que se deseja gerar.

            rows: quantidade de linhas do mapa que se deseja gerar.

        Retorna:
            Lista de coordenadas de distâncias
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
        Função para gerar coordenadas cúbicas no formato [x,y,z] para um grid
        hexagonal odd-r (linhas ímpares deslocadas para a direita) para uma
        quantidade de colunas e linha pré-determinado.
        Args:
            n_columns: número de colunas

            n_rows: número de linhas

        Retorna:
            coordenadas: lista[x, y, z]
        """
        x_coord = []
        y_coord = []
        z_coord = []
        for j in range(n_rows):
            for i in range(n_columns):
                x = i-(j-(j & 1))/2
                z = j
                y = -x -z

                # Colocar nas listas
                x_coord.append(x)
                y_coord.append(y)
                z_coord.append(z)

        coordinates = np.column_stack([x_coord, y_coord, z_coord])
        return coordinates

    def cube_distance(self, a, b):
        """
        Calcula a distância euclideana entre duas coordenadas cúbicas
        Args:
            a: Primeira coordenada cúbica [x,y,z]
            b: Segunda coordenada cúbica [x,y,z]

        Returns:

            Distância manhatan entre as coordenadas.
        """
        return ((abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])) / 2)**2

    def oddr_to_cube(self, col, row):
        """
        Transforma as coordenadas de retangulares em cúbicas.

        Args:
            coord: Coordenada que deseja transformar.

        Returns:
            Coordenada cúbica no formato [x,y,z]
        """

        x = col - (row - (row & 1)) / 2
        z = row
        y = -x-z
        return [x, y, z]


    def generate_rec_lattice(self, n_rows, n_columns):
        """
        Gera as coordenadas xy dos BMUs para um grid retangular.

        Args:
            n_rows: Número de linhas do mapa kohonen.
            n_columns: Número de colunas no mapa kohonen.

        Returns:
            Coordenadas do formato [x,y] para os bmus num grid retangular.
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
        Gera as coordenadas xy dos BMUs para um grid hexagonal odd-r.
        Args:
            n_rows:Número de linhas do mapa kohonen.
            n_columns:Número de colunas no mapa kohonen.

        Returns:
            Coordenadas do formato [x,y] para os bmus num grid hexagonal.

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
