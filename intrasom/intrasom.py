# Importações externas
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

# Plots
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Rectangle
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib as mpl


# Importações internas
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

        Constrói um objeto para treinamento de SOM, com os parâmetros de dados,
        de mapa e de tipo de treinamento.

        Args:
            data: Dados de entrada, representados por uma matriz de n linhas, como 
                amostras ou instâncias; e m colunas, como variáveis. São aceitos os 
                formatos dataframe ou ndarray (na escolha pelo formato ndarray, os 
                parâmetros `component_names` e `sample_names` podem ser preenchidas).


            mapsize: Tupla/lista definindo o tamanho do mapa SOM no formato 
                (colunas, linhas). Se um número inteiro é provido, ele é considerado 
                o número de neurônios. Por exemplo, para um mapa de 144 neurônios, 
                será criado automaticamente um mapa SOM (12, 12). Para o 
                desenvolvimento da periodicidade das grades hexagonais não é possível 
                a criação de mapas SOM com número ímpar de linhas. Assim, em escolhas 
                de tamanhos de mapas com números ímpares de linhas, será admitido de 
                forma automática o número inteiro par imediatamente inferior ao escolhido. 
                Se nenhum número for inserido, será considerado o tamanho fornecido pela 
                função heurística definida em `expected_mapsize()`.

            mask: mascara para os os valores nulos. Exemplo: -9999

            mapshape: formato da topologia do som. Exemplo: "planar" ou "toroid"

            lattice: tipo de lattice. Exemplo: "hexa"

            normalization: objeto para calculo de normalização. "var" ou "none"

            initialization: Método utilizado para a inicialização do som. Opções: 
                "pca" (apenas para bases de dados completas, sem NaN; não funciona com 
                dados faltantes) ou "random"

            neighborhood: objeto para cálculo de vizinhança.  Exemplot: "gaussian"

            training: modo de Treinamento. Opções: "batch"

            name: nome utilizado para identificar o objeto som, irá nomear o
                arquivo salvo ao fim do treinamento.

            component_names: lista com os rótulos das variáveis usadas no
                treinamento. Caso não seja fornecido, será criada
                automaticamente uma lista no formato:
                [Variável 1, Variável 2,...].

            unit_names: lista com os rótulos associados as unidades das
                variáveis de treinamento. Caso não seja fornecido, será criado
                automaticamente uma lista de unidades no estilo:
                [Unidade <variável1>, Unidade <variável2>,...].

            sample_names: lista com o nome das amostras. Caso não seja
                fornecida, será criada automaticamente um lista no formato:
                [Amostra 1, Amostra 2,...].

            missing: Valor booleano que deverá ser preenchido caso o banco de
                dados tenha valores faltantes (NaN). Para o treinamento do tipo
                Bruto ocorre uma procura pelos BMU utilizando uma função de
                calculo de distâncias com dados faltantes e a atualização do 
                codebook é feita com o preenchimento dos dados faltantes por 0.
                Na etapa de ajuste fino esse processo é repetido caso o parâmetro
                previous_epoch esteja sinalizado como Falso, ou há uma substituição
                dos valores vazios pelos valores calculados para essas células na
                época anterior de treinamento caso o parâmetro previous_epoch tenha
                sido sinalizado como Verdadeiro.Para que haja liberdade de 
                deslocamento dos vetores com dados faltantes ao longo do mapa 
                Kohonen, um fator aleatório de regularização sobre esse preenchimento 
                é gerado. Esse fator decai ao longo do treinamento em função do 
                decaimento do raio de busca para atualização de bmus. Esse fator 
                pode ser observado juntamente com a quantização do erro e o raio de 
                busca na barra iterativa de treinamento.

            pred_size: para o treinamento semi-supervisionado de bases de dados,
                se indica a colocação das colunas de rótulos de treinamento nas
                últimas posições do DataFrame e se indique aqui qual a
                quantidade das colunas rotuladas, para que a utilização da
                função de projeção project_nan_data() possa ser utilizada com
                dados não rotulados.

        Retorna:
            Objeto SOM com todos os seus métodos e atributos herdados.

        """
        # Aplicar a normalização caso essa esteja definida
        if normalization:
            normalizer = NormalizerFactory.build(normalization)
        else:
            normalizer = None

        # Construir o objeto de calculo de vizinhança de acordo com a função de
        # vizinhança especificada
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
        Função para carregamento de dados treinados. Necessário o acesso aos
        dataframes de entrada, dos neurônios treinados, assim como o arquivo de
        parâmetros salvo na finalização do processo de treinamento. Como
        demonstrado no notebook de exemplo de uso do código.

        Args:
            data: dados de entrada, representados por uma matriz de n linhas e
                m colunas como variáveis. Aceita o formato dataframe ou ndarray
                (caso seja ndarray as variáveis component_names e sample_names
                podem ser preenchidas).

            trained_neurons: o dataframe de neurônios treinados que é gerado ao fim 
                do treinamento SOM.

            params: JSON de parâmetros gerado ao fim do treinamento SOM.

        Retorna:
            Objeto SOM treinado e com todos os seus métodos e atributos
                herdados.
        """
        print("Carregando dados...")
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

        # Mascara para valores faltantes
        self.mask = mask

        # Checar tipo de entrada e preencher os atributos internos
        print("Carregando dataframe")
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
                self._sample_names = np.array(sample_names) if sample_names else np.array([f"Amostra_{i}" for i in range(1,data.shape[0]+1)])
            else:
                self._sample_names = np.array(sample_names) if sample_names else np.array([f"{i}" for i in range(1,data.shape[0]+1)])
        else:
            print("É aceito somente os tipos DataFrame ou Ndarray como\
             entradas para o IntraSOM")

        # Preencher os atributos não dependentes de tipo
        print("Normalizando")
        self._normalizer = normalizer
        self._normalization = normalization
        self._dim = data.shape[1]
        self._dlen = data.shape[0]
        self.pred_size = pred_size
        self.name = name
        self.missing = missing
        if self.missing == False:
            if np.isnan(self._data).any():
                sys.exit("Base com dados faltantes, sinalizar na variável missing")
        print("Criando vizinhança")
        self.neighborhood = neighborhood
        self._unit_names = unit_names if unit_names else [f"Unidade {var}" for var in self._component_names]
        self.mapshape = mapshape
        self.initialization = initialization
        
        if mapsize:
            if mapsize[1]%2!=0:
                self.mapsize = (mapsize[0], mapsize[1]+1)
                print(f"O número de linhas não pode ser ímpar.\
                O tamanho do mapa foi modificado para: {self.mapsize}")
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

        # Preencher atributos dependentes do tipo de carregamento
        if load_param:
            print("Criando banco de dados faltantes")
            self.data_missing = {"indices":tuple(zip(*np.argwhere(np.isnan(self._data)))), 
                                 "nan_values":missing_imput}
            # Modificar nome
            self.name = self.name+"_loaded"
            # Alocando os bmus
            self._bmu = bmus
            print("Criando codebook...")
            self.codebook = Codebook(self.mapsize, self.lattice, self.mapshape)
            self.codebook.matrix = self._normalizer.normalize_by(self.data_raw, trained_neurons.iloc[:,7:].values)
            
            try:
                print("Carregando matriz de distâncias...")
                self._distance_matrix = np.load("Resultados/distance_matrix.npy")
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
                self._distance_matrix = np.load("Resultados/distance_matrix.npy")
                if self.mapsize[0]*self.mapsize[1] != self._distance_matrix.shape[0]:
                    self._distance_matrix = self.calculate_map_dist
            except:
                self._distance_matrix = self.calculate_map_dist

    # PROPRIEDADES DE CLASSE
    
    @property
    def get_data(self):
        """
        Função de propriedade de classe para retornar uma cópia dos dados de 
        entrada com os dados faltantes.
        """
        if self.missing:
            self._data[self.data_missing["indices"]] = np.nan
        else:
            pass
        return self._data
    
    @property
    def params_json(self):
        """
        Função de propriedade de classe parar gerar um arquivo JSON com os
        parâmetros de treinamento.
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
        # Criar o dicionário de propriedades de treinamento
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
            
        
        # Arrumar problemas de serialização
        dic = fix_serialize(dic)

        # Transformar em JSON
        json_params = json.dumps(dic)

        # Salvar o resultado dentro do diretório especificado
        f = open(f"Resultados/params_{self.name}.json","w")
        f.write(json_params)
        f.close()

    @property
    def component_names(self):
        """
        Retornar os nomes das variáveis.
        """
        return self._component_names

    @property  
    def calculate_map_dist(self):
        """
        Calcula as distâncias do grid, que serão usadas durante as etapas de
        treinamento e as retorna no formato de uma matriz de distâncias internas
        do grid.
        """
        blen = 50
        
        # Capturar o número de neurons de treinamento
        nnodes = self.codebook.nnodes

        # Cria uma matriz de zeros no formato nnós x nnós
        distance_matrix = np.zeros((nnodes, nnodes))
        
        # Itera sobre os nós e preenche a matriz de distância para cada nó,
        # através da função grid_dist
        print("Inicializando mapa...")
        
        """
        # Acelerar o codigo com processamento paralelo
        def chunk_distmat_fill(nodes):
            for i in tqdm(nodes, desc="Matriz\
                de distâncias", unit=" Neurons"):
                dist = self.codebook.grid_dist(i)
                distance_matrix[i:,i] = dist[i:]
                distance_matrix[i,i:] = dist[i:]
                del dist
        """

        
        for i in tqdm(range(nnodes), desc="Matriz\
            de distâncias", unit=" Neurons"):
            dist = self.codebook.grid_dist(i)
            distance_matrix[i:,i] = dist[i:]
            distance_matrix[i,i:] = dist[i:]
            del dist
        
        # Criar diretórios se não existirem
        path = 'Resultados'
        os.makedirs(path, exist_ok=True)
        # Salvar para que esse processo seja feito somente 1x
        np.save('Resultados/distance_matrix.npy', distance_matrix) 

        return distance_matrix

    @property
    def neuron_matrix(self):
        """
        Retorna a matriz de neurônios denormalizada. No formato de array com os
        valores dos vetores de cada neurônio.
        """
        # Diferenciar a forma de carregar caso se trate de um carregamento de
        # dados treinados
        norm_neurons = self._normalizer.denormalize_by(self.data_raw, self.codebook.matrix)

        # Set a threshold for values near zero
        threshold = 1e-6

        # Transform values near zero to zero
        transformed_neurons = np.where(np.abs(norm_neurons) < threshold, 0, norm_neurons)
        
        return transformed_neurons

    @property
    def neurons_dataframe(self):
        """
        Função para criar um dataframe dos pesos dos neurônios resultantes do treinamento. São
        retornados no formado de um DataFrame dos neurônios e suas coordenadas
        retangulares e cúbicas.
        """

        # Criar dataframe
        neuron_df = pd.DataFrame(np.round(self.neuron_matrix,6),
                        index = list(range(1, self.neuron_matrix.shape[0]+1)),
                        columns=[f"B_{var}" for var in self._component_names])

        # Captar o número de colunas e linhas do mapa criado
        rows = self.mapsize[1]
        cols = self.mapsize[0]

        # Criar Coordenadas retangulares e cúbicas
        rec_coordinates = self._generate_rec_lattice(cols, rows)
        cub_coordinates = self._generate_oddr_cube_lattice(cols, rows)
        
        # Escala
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

        # Criar colunas
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
        Função para criar um dataframe com os BMUS e seus valores associados a
        cada vetor de entrada.
        """
        # Resgatar o dataframe de bmus
        bmu_df = self.neurons_dataframe
        bmus = self._bmu[0].astype(int)

        results_df = bmu_df.iloc[bmus,:]

        # Inserir o erro de quantização para cada vetor
        results_df.insert(1, "q-error", self.calculate_quantization_error_expanded)

        # Mudar index com o nome das amostras
        results_df.set_index(self._sample_names, inplace=True)

        # Regularizar o tipo de dado
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
        Função para criar um sumário do treinamento e salvar no formato .txt.
        """

        # Dicionário para tornar os termos mais explicativos
        dic_params={
            "var":"Variance",
            "toroid":"Toroid",
            "hexa":"Hexagonal",
            "random":"Randomic",
            "gaussian":"Gaussian",
            True:"Yes"
        }

        # Abrir um arquivo de texto
        text_file = open(f"Intrasom_report_{self.name}.txt", mode="w", encoding='utf-8')

        # Escrever as linhas de texto
        # Variáveis de projeto
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

        # Parâmetros de inicialização
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

        # Parâmetros de treinamento
        text_file.write(f"Training Parameters:\n")
        text_file.write(f"\n")
        text_file.write(f"Brute Training:\n")
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

        # Parâmetros de qualidade de treinamento
        text_file.write(f"Training Evaluation:\n")
        text_file.write(f"\n")
        text_file.write(f"Quantization Error: {round(self.calculate_quantization_error, 4)}\n")
        text_file.write(f"Topographic Error: {round(self.topographic_error, 4)}\n")
        text_file.close()
        print("Training Report Created")


    # MÉTODOS DE CLASSE
    def imput_missing(self, save=True, round_values=False):
        """
        Retorna os dados com os valores imputados nas células de entrada vazias.

        Args:
            save: valor booleano para indicar se o arquivo criado será ou não
            salvo dentro do diretório [Imputação].

        Retorna:
            DataFrame com dos dados de entrada com celulas vazias imputadas
                pelos seus respectivos BMU.
        """
        def minimum_decimal_places(array):
            """
            Função para descobrir o número de casas decimais em cada coluna de um array.
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
        
        # Captar os dados
        data = self.get_data
        data_folder = tempfile.mkdtemp()
        data_name = os.path.join(data_folder, 'data')
        dump(data, data_name)
        data = load(data_name, mmap_mode='r+')

        # Preencher
        data[self.data_missing["indices"]] = self.data_missing["nan_values"]

        # Denormalizar
        data = self._normalizer.denormalize_by(self.data_raw, data)
        if round_values:
            # Arredondar para o mínimo de casas decimais de cada coluna de treinamento
            min_dec = minimum_decimal_places(self.data_raw)

            # Iterate over each column and round the values with the corresponding decimal
            for i in range(data.shape[1]):
                data[:, i] = np.round(data[:, i], min_dec[i])
        else:
            data = np.round(data, decimals=6)

        # Substituir os -0 po 0
        data = np.where((data == -0) | (data == -0.0), 0, data)

        # Criar dataframe
        imput_df = pd.DataFrame(data, columns = self._component_names, index = self._sample_names)
        if save:
            # Criar diretórios se não existirem
            path = 'Imputação'
            os.makedirs(path, exist_ok=True)
            
            # Salvar
            imput_df.to_excel("Imputação/Dados_imputados.xlsx")
            imput_df.to_csv("Imputação/Dados_imputados.csv")

        return imput_df

    def project_nan_data(self,
                        data_proj,
                        with_labels=False,
                        sample_names=None,
                        save = True):
        """
        Função para projetar dados novos no modelo treinado, mesmo na presença 
        de valores faltantes.

        Args:
            data_proj: Dados se deseja projetar no modelo. Podem estar no
                formato DataFrame ou numpy ndarray.

            with_labels: Valor booleano para indicar se os dados tem as colunas
                de rotulos (modelo de classificação semi-supervisionada) ou não.
                Caso sinalizado como verdadeiro, irá considerar o número de colunas
                especificado em pred_size definido na criação do objeto SOM como
                as colunas de rótulos.
            
            save: Valor booleano para indicar o salvamento dos dados projetados,
                esses dados serão salvos no diretório "Projecoes" criado automaticamente.

        Returns:
            DataFrame com os BMUs representantes de cada vetor de entrada 
            projetado no mapa treinado.
        """
        
        # Checar formatos para adptação
        if isinstance(data_proj, pd.DataFrame):
            sample_names = sample_names if sample_names is not None else data_proj.index.values
            data_proj = data_proj.values
        elif isinstance(data_proj, np.ndarray):
            data_proj = data_proj
            sample_names = sample_names if sample_names is not None else [f"Amostra_proj_{i}" for i in range(1,data_proj.shape[0]+1)]
        else:
            print("Somente os formatos DataFrame e Ndarray são aceitos como entrada")

        # Checar presença de rótulos de treinamento nos dados a serem projetados
        if with_labels:
            # Retirar as variaveis label dos dados
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

        # Encontrar o BMU para esses novos dados
        bmus = self._find_bmu(data_proj, project=True)
        bmus_ind = bmus[0].astype(int)

        # Resgatar o dataframe de bmus
        bmu_df = self.neurons_dataframe

        # Criar dataframe para projecao
        projected_df = bmu_df.iloc[bmus_ind,:]

        projected_df.set_index(np.array(sample_names), inplace=True)

        projected_df = projected_df.astype({"BMU": int,
                                            "Ret_x": int,
                                            "Ret_y": int,
                                            "Cub_x": int,
                                            "Cub_y": int,
                                            "Cub_z": int
                                           })
        # Salvar
        if save:
            # Criar pasta de resultados
            try:
                os.mkdir("Projecoes")
            except:
                pass

            # Salvar
            projected_df.to_excel("Projecoes/Dados_projetados.xlsx")
            projected_df.to_csv("Projecoes/Dados_projetados.csv")

        return projected_df

    def denorm_data(self, data):
        """
        Método de classe para denormalizar dados de acordo com a normalização
        feita para os dados de entrada.
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
        Método de classe para treinamento do objeto SOM.

        Args:
            n_job: número de trabalhos para usar e paralelizar o treinamento.

            shared_memory: bandeira para ativar a memória compartilhada.

            train_rough_len: Numero de iterações durante o treinamento bruto.

            train_rough_radiusin: Raio inicial de busca de BMUs durante o
                treinamento bruto.

            train_rough_radiusfin: Raio final de busca de BMUs durante o
                treinamento bruto.

            train_finetune_len: Número de iterações durante o treinamento fino.

            train_finetune_radiusin: Raio inicial de busca de BMUs durante o
                treinamento fino.

            train_finetune_radiusfin: Raio final de busca de BMUs durante o
                treinamento fino.

            train_len_factor: Fator que multiplica os valores de extensão do
                treinamento (rought, fine, etc)

            maxtrainlen: Valor máximo de interações desejado.
                Default: np.Inf (infinito).

        Retorna:
            Objeto som treinado segundo os parâmetros escolhidos.

        """
        # Criar atributos de classe relacionados ao Treinamento
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

        print("Iniciando Treinamento")

        # Aplicar o tipo de inicialização escolhida
        if self.missing:
            if self.load_param:
                self.codebook.pretrain()
            elif self.initialization == 'random':
                self.codebook.random_initialization(self.get_data)
            elif self.initialization == 'pca':
                print("Ainda não implementado")
        else:
            if self.load_param:
                self.codebook.pretrain()
            elif self.initialization == 'random':
                self.codebook.random_initialization(self.get_data)
            elif self.initialization == 'pca':
                self.codebook.pca_linear_initialization(self.get_data)

        # Aplicar o tipo de treinamento escolhido
        if self.training == 'batch':
            print("Treinamento Bruto:")
            self.actual_train = "Bruto"
            self.rough_train(njob=n_job,
                             shared_memory=shared_memory,
                             trainlen=train_rough_len,
                             radiusin=train_rough_radiusin,
                             radiusfin=train_rough_radiusfin,
                             train_len_factor=train_len_factor,
                             maxtrainlen=maxtrainlen)
            
            print("Ajuste Fino:")
            self.actual_train = "Fino"
            self.finetune_train(njob=n_job,
                                shared_memory=shared_memory,
                                trainlen=train_finetune_len,
                                radiusin=train_finetune_radiusin,
                                radiusfin=train_finetune_radiusfin,
                                train_len_factor=train_len_factor,
                                maxtrainlen=maxtrainlen)
            
            
            if self.save:
                print("Salvando...")

                # Criar diretórios se não existirem
                path = 'Resultados'
                os.makedirs(path, exist_ok=True)

                # Salvar os resultados
                if dtypes == "xlsx_csv":
                    self.results_dataframe.to_excel(f"Resultados/{self.name}_resultados.xlsx")
                    self.neurons_dataframe.to_excel(f"Resultados/{self.name}_neurons.xlsx")
                    self.results_dataframe.to_csv(f"Resultados/{self.name}_resultados.csv")
                    self.neurons_dataframe.to_csv(f"Resultados/{self.name}_neurons.csv")
                elif dtypes == "xlsx":
                    self.results_dataframe.to_excel(f"Resultados/{self.name}_resultados.xlsx")
                    self.neurons_dataframe.to_excel(f"Resultados/{self.name}_neurons.xlsx")
                elif dtypes == "csv":
                    self.results_dataframe.to_csv(f"Resultados/{self.name}_resultados.csv")
                    self.neurons_dataframe.to_csv(f"Resultados/{self.name}_neurons.csv")
                elif dtypes == "parquet":
                    self.results_dataframe.to_parquet(f"Resultados/{self.name}_resultados.parquet")
                    self.neurons_dataframe.to_parquet(f"Resultados/{self.name}_neurons.parquet")
                else:
                    print("Tipo de salvamento escolhido está incorreto.")
            if self.summary:
                self.training_summary

            #Criar json de parâmetros de treinamento
            self.params_json

            print("Treinamento finalizado com sucesso.")

        elif self.training == 'seq':
            print("Ainda não implementado")
        else:
            print("O tipo de treinamento escolhido não está na lista aceitável: 'batch' ou 'seq'")

    def _calculate_ms_and_mpd(self):
        """
        Função para calcular o mpd e o ms. O mpd=neurônios/dados que segundo
        Vesanto (2000) são de 10 x mpd para o treinamento bruto e 40 x mpd para
        o treinamento de ajuste fino. Entretanto esses fatores serão
        considerados 20x e 60x para justificar a convergência de dados
        faltantes. Esses valores ainda não foram testados e podem ser
        otimizados no futuro.

        mpd = numero de nós do mapa kohonen dividido pelo número de amostras de
            entrada (linhas de dados)
        ms = maior dimensão do mapa kohonen
        """

        mn = np.min(self.codebook.mapsize)  # Menor dimensão do mapa
        max_s = max(self.codebook.mapsize[0], self.codebook.mapsize[1])  # Maior dimensão do mapa

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
        Método para implementação do treinamento bruto do objeto SOM.

        Args:
            njob: número de trabalhos em paralelo.

            shared_memory: uso de memória compartilhada.

            trainlen: número de iterações do treinamento bruto. Caso não seja
                preenchida será calculado o valor definido por:
                20x mpd (neurônios/dados).

            radiusin: Raio inicial do treinamento bruto. Caso não especificado
                se utilizará o valor ms/3 (ms-maior dimensão do mapa de
                treinamento).

            radiusfin: Raio inicial do treinamento bruto. Caso não especificado
                se utilizará o valor radiusin/6.

            trainlen_factor: fator que multiplicará o quantidade de épocas de
                treinamento.

            maxtrainlen: tamanho máximo de iterações permitidas para
                treinamento.
        """

        ms, mpd = self._calculate_ms_and_mpd()
        trainlen = min(int(np.ceil(30 * mpd)), maxtrainlen) if not trainlen else trainlen
        trainlen = int(trainlen * train_len_factor)

        # Definição automática dos valores de raio, caso não tenham sido definidos.
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
        Método para implementação do treinamento de ajuste fino do objeto SOM.

        Args:
            njob: número de trabalhos em paralelo.

            shared_memory: uso de memória compartilhada.

            trainlen: número de iterações do treinamento de ajuste fino. Caso
                não seja preenchida será calculado o valor definido por:
                60x mpd (neurônios/dados).

            radiusin: Raio inicial do treinamento de ajuste fino. Caso não
                especificado se utilizará o valor ms/12 (ms-maior dimensão do
                mapa de  treinamento).

            radiusfin: Raio inicial do treinamento bruto. Caso não especificado
                se utilizará o valor radiusin/25.

            trainlen_factor: fator que multiplicará o quantidade de épocas de
                treinamento.

            maxtrainlen: tamanho máximo de iterações permitidas para
                treinamento.
        """

        ms, mpd = self._calculate_ms_and_mpd()

        # Definição automática dos valores de raio, caso não tenham sido definidos.
        if self.initialization == 'random':
            trainlen = min(int(np.ceil(50 * mpd)), maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, ms / 8.) if not radiusin else radiusin  # do raio final no treinamento rough
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
        Método para implementação do treinamento em batch.

        Args:
            trainlen: número de iterações completas de treinamento.

            radiusin: raio inicial do treinamento.

            radiusfin: raio final do treinamento.

            njob: número de trabalhos em paralelo.

            shared_memory: uso de memória compartilhada.

        Returns:
            Retorna o resultado do treinamento em batch (atualizando as
            variáveis de classe) para os parâmetros selecionados.
        """
        # Achar o range de raios entre o raio inicial e final com a quantidade de loop especificada pelo trainlen
        radius = np.linspace(radiusin, radiusfin, trainlen)

        bmu = None

        # Processso de treinamento para inputs com dados completos
        if self.missing == False:     
            if shared_memory:
                data = self.get_data
                data_folder = tempfile.mkdtemp()
                data_name = os.path.join(data_folder, 'data')
                dump(data, data_name)
                data = load(data_name, mmap_mode='r')

            else:
                data = self.get_data
                

            #Barra de Treinamento
            pbar = tqdm(range(trainlen), mininterval=1)
            for i in pbar:
                if self.bootstrap:
                    # Passar todos os dados na ultima epoca de treinamento, para garantir que
                    # todos os vetores tenham BMU
                    if self.actual_train == "Fino" and self.train_finetune_len == i:
                        bootstrap_i = np.arange(0, self._dlen)
                    else:
                        # Criar indices bootstrap para amostrar do array de treinamento
                        bootstrap_i = np.sort(
                            np.random.choice(
                                np.arange(0, self._dlen, 1), int(self.bootstrap_proportion * self._dlen), 
                                replace=False))
                        
                    # Define a vizinhança para cada raio especificado
                    neighborhood = self.neighborhood.calculate(
                        self._distance_matrix, radius[i], self.codebook.nnodes)

                    # Encontra o BMU para os dados.
                    bmu = self._find_bmu(data[bootstrap_i], njb=njob)

                    # Atualiza os BMUs com os dados
                    self.codebook.matrix = self._update_codebook_voronoi(data[bootstrap_i],
                                                                         bmu,
                                                                         neighborhood)

                    # X2 é parte da distancia euclideana usada durante o encontro dos bmus
                    # para cada linha de dado. Como é um valor fixo ele pode ser ignorado durante
                    # o encontro dos bmus para cada dado de entrada, mas é necessário para o calculo do 
                    # da quantificacao do erro.
                    fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(data[bootstrap_i], nan=0.0), np.nan_to_num(data[bootstrap_i], nan=0.0))

                    # Atualização da barra de progresso
                    pbar.set_description(f"Época: {i+1}. Raio:{round(radius[i],2)}. QE: {round(np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2)),4)}")


                    # Atualiza somente os bmus dos vetores que participaram dessa época de treinamento
                    self._bmu[:,bootstrap_i] = bmu
                
                # Treinamento sem bootstrap
                else:
                    # Define a vizinhança para cada raio especificado
                    neighborhood = self.neighborhood.calculate(
                        self._distance_matrix, radius[i], self.codebook.nnodes)

                    # Encontra o BMU para os dados.
                    bmu = self._find_bmu(data, njb=njob)

                    # Atualiza os BMUs com os dados
                    self.codebook.matrix = self._update_codebook_voronoi(data,
                                                                        bmu,
                                                                        neighborhood)

                    # X2 é parte da distancia euclideana usada durante o encontro dos bmus
                    # para cada linha de dado. Como é um valor fixo ele pode ser ignorado durante
                    # o encontro dos bmus para cada dado de entrada, mas é necessário para o calculo do 
                    # da quantificacao do erro.
                    fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(data, nan=0.0), np.nan_to_num(data, nan=0.0))

                    # Atualização da barra de progresso
                    pbar.set_description(f"Época: {i+1}. Raio:{round(radius[i],2)}. QE: {round(np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2)),4)}")
                    
                    # Atualizar os BMUs
                    self._bmu = bmu


        # Processo de treinamento para inputs com dados faltantes
        elif self.missing == True:
            if shared_memory:
                data = self.get_data
                data_folder = tempfile.mkdtemp()
                data_name = os.path.join(data_folder, 'data')
                dump(data, data_name)
                data = load(data_name, mmap_mode='r+')

            else:
                data = self.get_data


            #Barra de Progresso
            pbar = tqdm(range(trainlen), mininterval=1)
            for i in pbar:
                if self.bootstrap:
                    # Passar todos os dados na ultima epoca de treinamento, para garantir que
                    # todos os vetores tenham BMU
                    if i==0:
                        bootstrap_i = np.arange(0, self._dlen)
                    else:
                        # Criar indices bootstrap para amostrar do array de treinamento
                        bootstrap_i = np.sort(
                            np.random.choice(
                                np.arange(0, self._dlen, 1), int(self.bootstrap_proportion * self._dlen), 
                                replace=False))
                    
                    # Define a vizinhança para cada raio especificado
                    neighborhood = self.neighborhood.calculate(
                        self._distance_matrix, radius[i], self.codebook.nnodes)

                    # Encontra o BMU e atualiza os dados inputados para busca do BMU
                    # conforme o treinamento

                    # Apresenta matriz incompleta no treinamento bruto
                    if self.actual_train == "Bruto":
                        # Encontra os dados dos BMUs
                        #data = np.nan_to_num(data)
                        bmu = self._find_bmu(data[bootstrap_i], njb=njob)
                        
                        # Atualiza os pesos de acordo coma função de vizinhança especificada
                        self.codebook.matrix = self._update_codebook_voronoi(data[bootstrap_i],
                                                                             bmu,
                                                                             neighborhood, 
                                                                             missing=True)
                        
                        # Preencher com os valores de epocas anteriores para manter caso nao participe
                        # do batch atual
                        data[self.data_missing["indices"]] = self.data_missing["nan_values"]

                        #Preencher os locais de dados vazios com os valores encontrados nos bmus
                        # a cada iteração
                        nan_mask = np.isnan(self.data_raw[bootstrap_i])

                        for j in range(self._data[bootstrap_i].shape[0]):
                            bmu_index = bmu[0][j].astype(int)
                            # Inserir um componente de aleatoriedade e regularização na
                            # imputação dos dados durante o treinamento, para não
                            # convergir tão rapido
                            data[bootstrap_i[j]][nan_mask[j]] = self.codebook.matrix[bmu_index][nan_mask[j]]
 
                        # Atualizar dados faltantes
                        self.data_missing["nan_values"] = data[self.data_missing["indices"]]

                        # Apagar os dados na variavel data
                        data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)

                        if self.save_nan_hist:
                            # Adicionar no historico de processamento de nan
                            self.nan_value_hist.append(self.data_missing["nan_values"])

                        # Atualização da barra de progresso
                        QE = round(np.mean(np.sqrt(bmu[1])),4)
                        pbar.set_description(f"Época: {i+1}. Raio:{round(radius[i],2)}. QE: {QE}")

                    # Apresenta matriz inputada no treinamento fino
                    elif self.actual_train == "Fino":
                        if self.previous_epoch:
                            #preenche os valores faltantes no data com os dados da interacao anterior para busca BMU
                            data[self.data_missing["indices"]] = self.data_missing["nan_values"]

                            # Encontra os dados dos BMUs
                            bmu = self._find_bmu(data[bootstrap_i], njb=njob)
                            fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(data[bootstrap_i], nan=0.0), np.nan_to_num(data[bootstrap_i], nan=0.0))


                            #Preencher os locais de dados vazios com os valores encontrados nos bmus
                            # a cada iteração
                            nan_mask = np.isnan(self.data_raw[bootstrap_i])

                            # Fator de regularização
                            reg = radius[i]/self.total_radius-1/self.total_radius

                            for j in range(self._data[bootstrap_i].shape[0]):
                                bmu_index = bmu[0][j].astype(int)
                                # Inserir um componente de aleatoriedade e regularização na
                                # imputação dos dados durante o treinamento, para não
                                # convergir tão rapido
                                data[bootstrap_i[j]][nan_mask[j]] = self.codebook.matrix[bmu_index][nan_mask[j]]*np.random.uniform(1-reg, 1+reg, np.sum(nan_mask[j]))

                            # Atualiza os pesos de acordo coma função de vizinhança especificada
                            self.codebook.matrix = self._update_codebook_voronoi(data[bootstrap_i],
                                                                                bmu,
                                                                                neighborhood)

                            # Atualizar dados faltantes
                            self.data_missing["nan_values"] = data[self.data_missing["indices"]]

                            # Apagar os dados na variavel data
                            data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)

                            if self.save_nan_hist:
                                # Adicionar no historico de processamento de nan
                                self.nan_value_hist.append(self.data_missing["nan_values"])

                            # Atualização da barra de progresso
                            QE = round(np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2)),4)
                            pbar.set_description(f"Época: {i+1}. Raio:{round(radius[i],2)}. QE: {QE}. Reg:{round(reg,2)}")
                        else:
                            # Encontra os dados dos BMUs
                            #data = np.nan_to_num(data)
                            bmu = self._find_bmu(data[bootstrap_i], njb=njob)
                            
                            # Atualiza os pesos de acordo coma função de vizinhança especificada
                            self.codebook.matrix = self._update_codebook_voronoi(data[bootstrap_i],
                                                                                bmu,
                                                                                neighborhood, 
                                                                                missing=True)
                            
                            # Preencher com os valores de epocas anteriores para manter caso nao participe
                            # do batch atual
                            data[self.data_missing["indices"]] = self.data_missing["nan_values"]

                            #Preencher os locais de dados vazios com os valores encontrados nos bmus
                            # a cada iteração
                            nan_mask = np.isnan(self.data_raw[bootstrap_i])

                            for j in range(self._data[bootstrap_i].shape[0]):
                                bmu_index = bmu[0][j].astype(int)
                                # Inserir um componente de aleatoriedade e regularização na
                                # imputação dos dados durante o treinamento, para não
                                # convergir tão rapido
                                data[bootstrap_i[j]][nan_mask[j]] = self.codebook.matrix[bmu_index][nan_mask[j]]
    
                            # Atualizar dados faltantes
                            self.data_missing["nan_values"] = data[self.data_missing["indices"]]

                            # Apagar os dados na variavel data
                            data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)

                            if self.save_nan_hist:
                                # Adicionar no historico de processamento de nan
                                self.nan_value_hist.append(self.data_missing["nan_values"])

                            # Atualização da barra de progresso
                            QE = round(np.mean(np.sqrt(bmu[1])),4)
                            pbar.set_description(f"Época: {i+1}. Raio:{round(radius[i],2)}. QE: {QE}")

                    if self.history_plot:
                        if i%2 == 0:
                            self.plot_umatrix(figsize = (5,3),
                                         hits = True,
                                         save = True,
                                         file_name = f"{self.actual_train}_epoca{i+1}",
                                         bmu=bmu)

                    # Atualiza somente os bmus dos vetores que participaram dessa época de treinamento
                    self._bmu[:, bootstrap_i] = bmu
                else:
                    # Define a vizinhança para cada raio especificado
                    neighborhood = self.neighborhood.calculate(
                        self._distance_matrix, radius[i], self.codebook.nnodes)

                    # Encontra o BMU e atualiza os dados inputados para busca do BMU
                    # conforme o treinamento

                    # Apresenta matriz incompleta no treinamento bruto
                    if self.actual_train == "Bruto":
                        # Encontra os dados dos BMUs
                        #data = np.nan_to_num(data)
                        bmu = self._find_bmu(data, njb=njob)

                        # Atualiza os pesos de acordo coma função de vizinhança especificada
                        self.codebook.matrix = self._update_codebook_voronoi(data,
                                                                            bmu,
                                                                            neighborhood, 
                                                                            missing=True)
                        
                        #Preencher os locais de dados vazios com os valores encontrados nos bmus
                        # a cada iteração
                        nan_mask = np.isnan(self.data_raw)

                        for j in range(self._data.shape[0]):
                            bmu_index = bmu[0][j].astype(int)
                            # Inserir um componente de aleatoriedade e regularização na
                            # imputação dos dados durante o treinamento, para não
                            # convergir tão rapido
                            data[j][nan_mask[j]] = self.codebook.matrix[bmu_index][nan_mask[j]]
                        
                        # Atualizar dados faltantes
                        self.data_missing["nan_values"] = data[self.data_missing["indices"]]

                        # Apagar os dados na variavel data
                        data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)
                        if self.save_nan_hist:
                            # Adicionar no historico de processamento de nan
                            self.nan_value_hist.append(self.data_missing["nan_values"])

                        # Atualização da barra de progresso
                        QE = round(np.mean(np.sqrt(bmu[1])),4)
                        pbar.set_description(f"Época: {i+1}. Raio:{round(radius[i],2)}. QE: {QE}")

                    # Apresenta matriz inputada no treinamento fino
                    elif self.actual_train == "Fino":
                        if self.previous_epoch:

                            #preenche os valores faltantes no data com os dados da interacao anterior para busca BMU
                            data[self.data_missing["indices"]] = self.data_missing["nan_values"]

                            # Encontra os dados dos BMUs
                            bmu = self._find_bmu(data, njb=njob)
                            fixed_euclidean_x2 = np.einsum('ij,ij->i', np.nan_to_num(data, nan=0.0), np.nan_to_num(data, nan=0.0))


                            #Preencher os locais de dados vazios com os valores encontrados nos bmus
                            # a cada iteração
                            nan_mask = np.isnan(self.data_raw)

                            # Fator de regularização
                            reg = radius[i]/self.total_radius-1/self.total_radius

                            for j in range(self._data.shape[0]):
                                bmu_index = bmu[0][j].astype(int)
                                # Inserir um componente de aleatoriedade e regularização na
                                # imputação dos dados durante o treinamento, para não
                                # convergir tão rapido
                                data[j][nan_mask[j]] = self.codebook.matrix[bmu_index][nan_mask[j]]*np.random.uniform(1-reg, 1+reg, np.sum(nan_mask[j]))
                            
                            # Atualiza os pesos de acordo coma função de vizinhança especificada
                            self.codebook.matrix = self._update_codebook_voronoi(data,
                                                                                bmu,
                                                                                neighborhood)

                            # Atualizar dados faltantes
                            self.data_missing["nan_values"] = data[self.data_missing["indices"]]

                            # Apagar os dados na variavel data
                            data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)

                            if self.save_nan_hist:
                                # Adicionar no historico de processamento de nan
                                self.nan_value_hist.append(self.data_missing["nan_values"])

                            # Atualização da barra de progresso
                            QE = round(np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2)),4)
                            pbar.set_description(f"Época: {i+1}. Raio:{round(radius[i],2)}. QE: {QE}. Reg:{round(reg,2)}")
                        else:
                            # Encontra os dados dos BMUs
                            #data = np.nan_to_num(data)
                            bmu = self._find_bmu(data, njb=njob)

                            # Atualiza os pesos de acordo coma função de vizinhança especificada
                            self.codebook.matrix = self._update_codebook_voronoi(data,
                                                                                bmu,
                                                                                neighborhood, 
                                                                                missing=True)
                            
                            #Preencher os locais de dados vazios com os valores encontrados nos bmus
                            # a cada iteração
                            nan_mask = np.isnan(self.data_raw)

                            for j in range(self._data.shape[0]):
                                bmu_index = bmu[0][j].astype(int)
                                # Inserir um componente de aleatoriedade e regularização na
                                # imputação dos dados durante o treinamento, para não
                                # convergir tão rapido
                                data[j][nan_mask[j]] = self.codebook.matrix[bmu_index][nan_mask[j]]
                            
                            # Atualizar dados faltantes
                            self.data_missing["nan_values"] = data[self.data_missing["indices"]]

                            # Apagar os dados na variavel data
                            data[self.data_missing["indices"]] = np.full(len(self.data_missing["indices"][0]), np.nan)
                            if self.save_nan_hist:
                                # Adicionar no historico de processamento de nan
                                self.nan_value_hist.append(self.data_missing["nan_values"])

                            # Atualização da barra de progresso
                            QE = round(np.mean(np.sqrt(bmu[1])),4)
                            pbar.set_description(f"Época: {i+1}. Raio:{round(radius[i],2)}. QE: {QE}")

                    if self.history_plot:
                        if i%2 == 0:
                            self.plot_umatrix(figsize = (5,3),
                                         hits = True,
                                         save = True,
                                         file_name = f"{self.actual_train}_epoca{i+1}",
                                         bmu=bmu)
                    # Atualizar bmus
                    self._bmu = bmu
            
            self.params_json


    def _find_bmu(self,
                 input_matrix,
                 njb=-1,
                 nth=1,
                 project=False):
        """
        Encontra os BMUs (Best Matching Units) para cada dado de input através
        da matriz de dados de entrada. De forma unificada paralelizando o
        cálculo ao invés de usar dado a dado e comparar com o codebook.

        Args:
            input_matrix: matriz numpy ndarray representando as amostras do
                input como linhas e as variáveis como colunas.

            njb: numero de jobs para a busca paralelizada. Default: -1 (Busca
                automática pelo número de cores de processamento)

            nth:

            project: valor booleano para o encontro de BMUs relacionados a uma
                base de projeção em um mapa já treinado.

        Returns:
            O BMU para cada dado de input no formato [[bmus],[distâncias]].
        """

        dlen = input_matrix.shape[0]
        if njb == -1:
            njb = cpu_count()
        y2 = np.einsum('ij,ij->i', self.codebook.matrix, self.codebook.matrix)
        
        
        pool = Pool(njb)

        # Cria objeto para achar bmu em pedaços de dado
        chunk_bmu_finder = self._chunk_based_bmu_find

        def row_chunk(part):
            return part * dlen // njb

        def col_chunk(part):
            return min((part + 1) * dlen // njb, dlen)
        
        # Separa pedaços dos dados de input input_matrix para serem analisados
        # pelo chunk_bmu_finder
        chunks = [input_matrix[row_chunk(i):col_chunk(i)] for i in range(njb)]


        if project:
            missing_proj = np.isnan(input_matrix).any()
            if missing_proj:
                # Mapeia os pedaços de dado e aplica o método chunk_bmu_finder em cada pedaço, achando os bmus para cada pedaço
                b = pool.map(lambda chk: chunk_bmu_finder(chk,
                                                          self.codebook.matrix[:, :input_matrix.shape[1]],
                                                          y2,
                                                          nth=nth, 
                                                          project=project,
                                                          missing=missing_proj),
                                                          chunks)
            else:
                # Mapeia os pedaços de dado e aplica o método chunk_bmu_finder em cada pedaço, achando os bmus para cada pedaço
                b = pool.map(lambda chk: chunk_bmu_finder(chk,
                                    self.codebook.matrix[:, :input_matrix.shape[1]], 
                                    y2=y2,
                                    project=project,
                                    nth=nth),
                             chunks)
                
        else:
            if self.missing:
                # Mapeia os pedaços de dado e aplica o método chunk_bmu_finder em cada pedaço, achando os bmus para cada pedaço
                b = pool.map(lambda chk: chunk_bmu_finder(chk,
                                                          self.codebook.matrix,
                                                          y2=y2,
                                                          nth=nth, 
                                                          missing=True),
                                                          chunks)
            else:
                # Mapeia os pedaços de dado e aplica o método chunk_bmu_finder em cada pedaço, achando os bmus para cada pedaço
                b = pool.map(lambda chk: chunk_bmu_finder(chk,
                                                          self.codebook.matrix, 
                                                          y2,
                                                          nth=nth),
                                                          chunks)
        pool.close()
        pool.join()

        # Organiza os chuncks de dados de BMU em um array [2,dlen] em que a
        # primeira linha tem os bmus e a segunda as distâncias
        bmu = np.asarray(list(itertools.chain(*b))).T
        del b
        return bmu


    def _update_codebook_voronoi(self, training_data, bmu, neighborhood, missing=False):
        """
        Método para atualizar os pesos de cada nó no codebook que pertence a
        vizinhança do BMU. Primeiro encontra o set Voronoi de cada nó. Precisa
        calcular uma matriz menor. Mais rápido do que o algorítmo clássico em
        batch, é baseado na implementação do algoritmo Som Toolbox para
        Matlab pela Universidade de Helsinky. Primeiramente implementado em
        Python pela biblioteca SOMPY.

        Args:
            training_data: matriz de vetores de entrada como linhas e variáveis
                como colunas.

            bmu: BMU para cada dado de entrada. Tem o formato
                [[bmus],[distâncias]].

            neighborhood: matriz representando a vizinhança de cada bmu.

        Returns:
            Um codebook atualizado que incorpora o aprendizado dos dados de
                entrada.

        """
        if missing:
            # Cria uma máscara para os valores faltantes no training_data e substitui por 0
           training_data[np.isnan(training_data)] = 0

        # Pega todos os números de bmus de cada linha de dado e coloca no
        # formato Int
        row = bmu[0].astype(int)

        # Todos os índices para as colunas
        col = np.arange(training_data.shape[0])

        # array com 1 repetido no tamanho das linhas de dado
        val = np.tile(1, training_data.shape[0])

        # Cria uma matrix esparsa (csr -> compressed sparsed row) com a chamada
        # csr_matrix((val, (row, col)), [shape=(nnodes, dlen)])
        P = csr_matrix((val, (row, col)), shape=(self.codebook.nnodes,
                                                 training_data.shape[0]))

        # Multiplica pelos dados de input para retornar uma matriz S com os
        # dados de input nos bmus
        S = P.dot(training_data)

        # Multiplica os valores da vizinhança pela matriz S com os valores de
        # input nos bmus
        nom = neighborhood.T.dot(S)

        # Conta quantas vezes cada BMU foi selecionado por um vetor de entrada
        nV = P.sum(axis=1).reshape(1, self.codebook.nnodes)

        # Multiplica a quantidade de vezes que o bmu foi selecionado pelos
        # valores de entrada pela função de vizinhança
        denom = nV.dot(neighborhood.T).reshape(self.codebook.nnodes, 1)

        # Divide os valores do nominador pelo denominador
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
        Encontra os BMUs correspondentes a matrix de dados de entrada.

        Args:
            input_matrix: uma matriz dos dados de entrada, representando os vetores
                de entrada nas linhas e as variáveis do vetor nas colunas. Quando a
                busca é paralelizada, a matriz de entrada pode ser uma sub-matriz de
                uma matriz maior.

            codebook: matriz de pesos para ser usada na busca dos BMUs.

            nth:

        Returns:
            Retorna os BMUS e as distâncias para a matriz ou sub-matriz de vetores
            de entrada. No formato [[bmu],[distância]].
        """
        def dist_by_type(codebook, ddata, missing, train_type=None):
            """
            Função para escolher o tipo de distãncia a ser calculada dependendo da presença de dados
            faltantes e/ou do estágio de treinamento.
            """
            if missing:
                if train_type == "nan_euclid" or train_type == "Projetado":
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

        # Quantidade de inputs (linhas) dos dados de entrada)
        dlen = input_matrix.shape[0]

        # Inicializa um array com dlen de linhas e duas colunas
        bmu = np.empty((dlen, 2))
        
        # Quantidade de nos de treinamento no codebook
        nnodes = codebook.shape[0]

        # Tamanho do batch
        blen = min(50, dlen)

        # Inicializador do loop while
        i0 = 0
        
        if missing:
            while i0 + 1 <= dlen:
                # Inicio da procura nos dados (linhas da matriz inputada)
                low = i0  

                # Fim da procura nos dados (linhas da matriz inputada) para esse batch
                high = min(dlen, i0 + blen)
                i0 = i0 + blen  # Atualizar o inicializador do loop

                # Recorte na matriz de input nessas amostras do batch.
                ddata = input_matrix[low:high + 1]

                if self.actual_train == "Bruto" or self.previous_epoch == False:
                    type_search = "nan_euclid"
                else:
                    type_search = "Fino"

                d = dist_by_type(codebook=codebook, 
                                 ddata=ddata, 
                                 missing=missing, 
                                 train_type = "Projetado" if project else type_search)

                # Encontrar os BMUs
                # Função para encontrar a posição dentro do array de distâncias que está
                # o menor valor (o BMU), e o designar como primeiro atributo da variável
                # BMU
                bmu[low:high + 1, 0] = np.argpartition(d, nth, axis=0)[nth - 1]

                # Função para pegar menor distância e designar como segundo atributo da
                # variável BMU
                bmu[low:high + 1, 1] = np.partition(d, nth, axis=0)[nth - 1]
                del ddata
        else:
            while i0 + 1 <= dlen:
                # Inicio da procura nos dados (linhas da matriz inputada)
                low = i0  

                # Fim da procura nos dados (linhas da matriz inputada) para esse batch
                high = min(dlen, i0 + blen)
                i0 = i0 + blen  # Atualizar o inicializador do loop

                # Recorte na matriz de input nessas amostras do batch.
                ddata = input_matrix[low:high + 1]

                d = dist_by_type(codebook=codebook, 
                                 ddata=ddata, 
                                 missing=missing)

                # Encontrar os BMUs
                # Função para encontrar a posição dentro do array de distâncias que está
                # o menor valor (o BMU), e o designar como primeiro atributo da variável
                # BMU
                bmu[low:high + 1, 0] = np.argpartition(d, nth, axis=0)[nth - 1]

                # Função para pegar menor distância e designar como segundo atributo da
                # variável BMU
                bmu[low:high + 1, 1] = np.partition(d, nth, axis=0)[nth - 1]
                del ddata

        return bmu
    
    @property
    def topographic_error(self):
        """
        Função que está no SOMPY, discordo dessa função, ela busca somente se o primeiro e segundo bmus que 
        melhor representam o vetor de entrada são vizinhos.
        """
        bmus2 = self._find_bmu(self.get_data, nth=2)
               
        dist_matrix_1 = self._distance_matrix
        topographic_error = 1-(np.array(
                [distances[bmu2] for bmu2, distances in zip(bmus2[0].astype(int), dist_matrix_1)]) > 4).mean()

        return topographic_error
    
    @property
    def calculate_topographic_error(self):
        """
        Cálculo do Erro Topográfico, que se trata de uma medida da qualidade da
        preservação da estrutura do espaço dos dados de entrada no mapa
        treinado. É calculado encontrando os neurônios de menor e segunda menor
        distância euclideana de cada vetor de entrada e avaliando se a
        preservação dessa vizinhança acontece no mapa de saída treinado. Se
        ocorre a preservação da vizinhança então se diz que houve preservação da
        topologia e o erro topográfico é baixo.
        Para cada vetor de entrada que não tem sua vizinhaça preservada é
        contado na proporção do erro topográfico, portanto é um erro que varia
        de 0 (todos os vetores de entrada mantiveram a vizinhança) a 1 (nenhum
        vetor de entrada preservou a sua vizinhança).

        """
        # Procurar em chunks para não ter gargalo de RAM
        #Iniciar o loop while
        i0 = 0
        
        # Tamanho do batch
        blen = 1000
        
        # Indices
        indices_before = np.zeros((self.data_raw.shape[0],2))
        indices_after = np.zeros((self.data_raw.shape[0],2))
        
        while i0+1 <= self._dlen:
            # Inicio da procura dos dados
            low = i0
            
            # Fim da procura dos dados
            high = min(self._dlen, i0+blen)
            
            #Recortar a matriz para esse batch
            ddata = self.data_raw[low:high + 1]
            
            #Matriz de distãncias pré-treinamento
            dist_before = nan_euclidean_distances(ddata, ddata)

            # Encontrar vetores mais próximos dos vetores de entrada
            argmin_before = np.argsort(dist_before, axis=0)[1,:]
            
            # Indices pareados de vetores mais próximos
            indices_before_batch = np.array([[i,j] for i,j in zip(np.arange(0, len(ddata), 1), argmin_before)])
            
            # Preencher vetor principal
            for i, j in zip(np.arange(low, high+1, 1), np.arange(0, blen+1, 1)):
                indices_before[i] = indices_before_batch[j]
            
            # Gerar coordenadas cúbicas para o mapa
            coordinates = self._generate_oddr_cube_lattice(self.mapsize[0], self.mapsize[1])
            cols, rows = self.mapsize[0], self.mapsize[1]
            
            # Criar vizinhança toroidal
            toroid_neigh = [[0, 0], [cols, 0], [cols, rows], [0, rows], [-cols, rows], [-cols, 0], [-cols, -rows], [0,-rows], [cols, -rows]]
            toroid_neigh = [self._oddr_to_cube(i[0], i[1]) for i in toroid_neigh]
            
            # Captar BMUs
            bmus = self._bmu[0][low:high + 1].astype(int)
            
            # Criar matriz vazia para preenchimento das distâncias manhatan dentro de um grid cúbico hexagonal
            dist_after = np.zeros([len(ddata),len(ddata)])
            for i in range(len(ddata)):
                for j in range(len(ddata)):
                    dist = int(min([self._cube_distance(coordinates[bmus[i]] + neig,coordinates[bmus[j]]) for neig in toroid_neigh]))
                    dist_after[j][i] = dist
                    
            # Encontrar hits mais próximos dos BMUs
            argmin_after = np.argsort(dist_after, axis=0)[1,:]
            
            # Indices pareados desses hits
            indices_after_batch = np.array([[i,j] for i,j in zip(np.arange(0, len(ddata), 1), argmin_after)])
            
            # Preencher vetor principal
            for i, j in zip(np.arange(low, high+1, 1), np.arange(0, blen+1, 1)):
                indices_after[i] = indices_after_batch[i]
            
            # Atualizar indice do loop
            i0=i0+blen
            
            del ddata

        # Erro Topográfico: 0 se se mantém a vizinhança e 1 se não se mantem
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
        Função para calcular a Matriz U de distâncias unificadas a partir da
        matriz de pesos treinada.

        Args:
            exapanded: valor booleano para indicar se o retorno será da matriz
                de distâncias unificadas resumida (média das distâncias dos 6
                bmus de vizinhança) ou expandida (todos os valores de distância)
        Retorna:
            Matriz de distâncias unificada expandida ou resumida.
        """
        # Função para encontrar distancia rápido
        def fast_norm(x):
            """
            Retorna a norma L2 de um array 1-D.
            """
            return np.sqrt(np.dot(x, x.T))

        # Matriz de pesos bmus
        weights = np.reshape(self.codebook.matrix, (self.mapsize[1], self.mapsize[0], self.codebook.matrix.shape[1]))

        # Busca hexagonal vizinhos
        ii = [[1, 1, 0, -1, 0, 1], [1, 0,-1, -1, -1, 0]]
        jj = [[0, 1, 1, 0, -1, -1], [0, 1, 1, 0, -1, -1]]

        # Inicializar Matriz U
        um = np.nan * np.zeros((weights.shape[0], weights.shape[1], 6))

        # Preencher matriz U
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                w_2 = weights[y, x]
                e = y % 2 == 0
                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                    if (x+i >= 0 and x+i < weights.shape[1] and y+j >= 0 and y+j < weights.shape[0]):
                        w_1 = weights[y+j, x+i]
                        um[y, x, k] = fast_norm(w_2-w_1)
        if expanded:
            # Matriz U expandida
            return um
        else:
            # Matriz U reduzida
            return np.nanmean(um, axis=2)
        
    
    def plot_umatrix(self,
                     figsize = (10,10),
                     hits = True,
                     save = True,
                     file_name = None,
                     file_path = False,
                     bmu=None):
        
        if file_name is None:
            file_name = f"Matriz_U_{self.name}"

        if hits:
            # Contagem de hits
            unique, counts = np.unique(bmu[0].astype(int), return_counts=True)

            # Normalizar essa contagem de 0.5 a 2.0 (de um hexagono pequeno até um
            #hexagono que cobre metade dos vizinhos).
            counts = minmax_scale(counts, feature_range = (0.5,2))

            bmu_dic = dict(zip(unique, counts))
            
        
        # Busca hexagonal vizinhos
        ii = [[1, 1, 0, -1, 0, 1], [1, 0,-1, -1, -1, 0]]
        jj = [[0, 1, 1, 0, -1, -1], [0, 1, 1, 0, -1, -1]]
        
        # Criar coordenadas
        xx = np.reshape(self._generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,0], (self.mapsize[1], self.mapsize[0]))
        yy = np.reshape(self._generate_hex_lattice(self.mapsize[0], self.mapsize[1])[:,1], (self.mapsize[1], self.mapsize[0]))
        
        # Matriz U
        um = self.build_umatrix(expanded = True)
        umat = self.build_umatrix(expanded = False)
        
        # Plotagem
        f = plt.figure(figsize=figsize)
        ax = f.add_subplot()
        ax.set_aspect('equal')
        
        # Normalizar as cores para todos os hexagonos
        norm = mpl.colors.Normalize(vmin=np.nanmin(um), vmax=np.nanmax(um))
        counter = 0
        
        for j in range(self.mapsize[1]):
            for i in range(self.mapsize[0]):
                # Hexagono Central
                hex = RegularPolygon((xx[(j,i)]*2,
                                      yy[(j,i)]*2),
                                     numVertices=6,
                                     radius=1/np.sqrt(3),
                                     facecolor= cm.jet(norm(umat[j][i])),
                                     alpha=1)#, edgecolor='black')

                ax.add_patch(hex)

                # Hexagono da Direita
                if not np.isnan(um[j, i, 0]):
                    hex = RegularPolygon((xx[(j, i)]*2+1,
                                          yy[(j,i)]*2),
                                         numVertices=6,
                                         radius=1/np.sqrt(3),
                                         facecolor=cm.jet(norm(um[j,i,0])),
                                         alpha=1)
                    ax.add_patch(hex)

                # Hexagono Superior Direita
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
                    
                #Plotar hits
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
                # Criar diretórios se não existirem
                path = 'Plotagens/U_matrix'
                os.makedirs(path, exist_ok=True)
                if hits:
                    f.savefig(f"Plotagens/U_matrix/{file_name}_with_hits.jpg",dpi=300, bbox_inches = "tight")
                else:
                    f.savefig(f"Plotagens/U_matrix/{file_name}.jpg",dpi=300, bbox_inches = "tight")
    
    def rep_sample(self, save=False, project=None):
        """
        Retorna um dicionário contendo as amostras representativas para cada neurônio de melhor correspondência 
        (BMU) do mapa auto-organizável (SOM).

        Args: 
            save (bool, optional): Indica se os resultados devem ser salvos em um arquivo de texto. 
            O padrão é False.

        Returns:
            dict: Um dicionário onde as chaves são os BMUs e os valores são as amostras 
            representativas associadas a cada BMU, em ordem de representatividade.
        """
        if project is not None:
            som_bmus = np.concatenate((self._bmu[0].astype(int),np.array(project.BMU.values-1)))
            sample_names = np.concatenate((np.array(self._sample_names), np.array(project.index.values)))
            data = np.concatenate((self.get_data, self.data_proj_norm), axis=0)
        else:
            som_bmus = self._bmu[0].astype(int)
            sample_names = self._sample_names
            data = self.get_data

        # Dicionário de labels com as amostras
        dic = {}
        for key, value in zip(som_bmus, sample_names):
            if key in dic:
                if isinstance(dic[key], list):
                    dic[key].append(value)
                else:
                    dic[key] = [dic[key], value]
            else:
                dic[key] = value

        # Dicionário de indices das amostras em cada BMU
        dic_index = {}
        for key, index in zip(som_bmus, range(len(sample_names))):
            if key in dic_index:
                if isinstance(dic_index[key], list):
                    dic_index[key].append(index)
                else:
                    dic_index[key] = [dic_index[key], index]
            else:
                dic_index[key] = index
        
        # Reorganização do dicionário pela ordem de distâncias
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
            name = "Amostras_representativas_projetadas" if project is not None else "Amostras_representativas"
            with open(f'Resultados/{name}.txt', 'w', encoding='utf-8') as file:
                for key, value in rep_samples_dic.items():
                    if isinstance(value, list):
                        value = ', '.join(value)
                    file.write(f'BMU {key+1}: {value}\n')

        return rep_samples_dic


    def _expected_mapsize(self, data):
        """
        Retorna o tamanho esperado do mapa com base na função eurística definida por
        Vessanto et al (2000) definida a seguir: 5 x sqrt(M).

        Args:
            data: os dados de entrada para o treinamento som.

        """
        expected = round(np.sqrt(5*np.sqrt(data.shape[0])))

        if expected%2!=0:
            row_expec = expected+1
        else:
            row_expec = expected

        return (expected, row_expec)  


    def _generate_hex_lattice(self, n_columns, n_rows):
        """
        Gera as coordenadas xy dos BMUs para um grid hexagonal odd-r. (Colunas
        ímpares deslocadas para a direita)

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


    def _generate_rec_lattice(self, n_columns, n_rows):
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


    def _oddr_to_cube(self, col, row):
        """
        Transforma as coordenadas de retangulares em cúbicas.

        Args:
            col: coordenada da coluna que deseja transformar.

            row: coordenada da linha que deseja transformar.

        Returns:
            Coordenada cúbica no formato [x,y,z]
        """

        x = col - (row - (row & 1)) / 2
        z = row
        y = -x-z
        return [x, y, z]


    def _cube_distance(self,a, b):
        """
        Calcula a distância manhatan entre duas coordenadas cúbicas.
        Args:
            a: Primeira coordenada cúbica no formato [x,y,z]

            b: Segunda coordenada cúbica no formato [x,y,z]

        Returns:
            Distância manhatan entre as coordenadas.
        """
        return (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])) / 2


    def _generate_oddr_cube_lattice(self, n_columns, n_rows):
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
                x_coord.append(int(x))
                y_coord.append(int(y))
                z_coord.append(int(z))

        coordinates = np.column_stack([x_coord, y_coord, z_coord])
        return coordinates




# silencilar logging do matplotlib
import logging
import sys
logging.getLogger('mtb.font_manager').disabled = True
logging.disable(sys.maxsize)

import warnings

# Ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
