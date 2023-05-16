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
from tqdm.notebook import tqdm

# Plots
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Rectangle
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib as mpl


# Importações internas
from .codebook import Codebook
from .object_functions import NeighborhoodFactory, NormalizerFactory
#from .normalization import NormalizerFactory
#from .visualization import UmatrixFactory


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
              pred_size=0):
        """

        Constrói um objeto para treinamento de SOM, com os parâmetros de dados,
        de mapa e de tipo de treinamento.

        Args:
            data: dados para serem clusterizados, representados por uma matriz
                de n linhas, como dados de entrada e m colunas como variáveis.
                Aceitando o formato dataframe ou ndarray (caso seja ndarray as
                variáveis component_names e sample_names podem ser preenchidas).


            mapsize: tupla/lista definindo as dimensões do mapa som. Se um
                número inteiro é provido ele é considerado o número de nós. Se
                nenhum número for inserido, será considerado o tamanho fornecido
                pela função heurística definida em expected_mapisize().

            mask: mascara para os os valores nulos. Exemplo: -999

            mapshape: formato da topologia do som. "planar" ou "toroid"

            lattice: tipo de lattice. "rect" ou "hexa"

            normalization: objeto para calculo de normalização. "var" ou "none"

            initialization: método utilizado para a inicialização do som.
                Opções: "pca" (Falta implementar para dados com NaN) ou "random"

            neighborhood: objeto para cálculo de vizinhança.  Opções: "gaussian"
                ou "bubble"

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
                batch ele vai preencher os dados faltantes pelos bmus na
                primeira iteração, atualizar os bmus e na segunda iteração
                preencher os dados faltantes novamente com os bmus da epóca
                anterior e se repete o processo até o fim do treinamento. Para
                que haja liberdade de deslocamento dos vetores com dados
                faltantes ao longo do mapa Kohonen, um fator aleatório de
                regularização sobre esse preenchimento é gerado. Esse fator
                decai ao longo do treinamento em função do decaimento do raio de
                busca para atualização de bmus. Esse fator pode ser observado
                juntamente com a quantização do erro e o raio de busca na barra
                iterativa de treinamento.

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

        return SOM(data,
                    neigh_calc,
                    normalizer,
                    normalization,
                    mapsize,
                    mask,
                    mapshape,
                    lattice,
                    initialization,
                    training,
                    name,
                    component_names,
                    unit_names,
                    sample_names,
                    missing,
                    pred_size)

    @staticmethod
    def load_som(data,
             trained_neurons,
             params):
        """
        Função para carregamento de dados treinados. Necessário o acesso aos
        dataframes de entrada, de resultados e de bmus, assim como o arquivo de
        parâmetros salvo na finalização do processo de treinamento.  Como
        demonstrado no notebook de exemplo de uso do código.

        Args:
            data: dados de entrada, representados por uma matriz de n linhas e
                m colunas como variáveis. Aceita o formato dataframe ou ndarray
                (caso seja ndarray as variáveis component_names e sample_names
                podem ser preenchidas).

            bmus: o dataframe de bmus gerado ao fim do treinamento SOM.

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
                self.data_raw[self.data_raw==self.mask]=np.nan
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
        self.reg_hist = []
        self.nan_value_hist = []

        # Preencher atributos dependentes do tipo de carregamento
        if load_param:
            print("Criando banco de dados faltantes")
            self.data_missing = {"indices":tuple(zip(*np.argwhere(np.isnan(self._data)))), 
                                 "nan_values":None}
            # Para acessar preencher os dados faltantes basta:
            # nan_array[tuple(zip(*indices))] = nan_values
            # Modificar nome
            self.name = self.name+"_loaded"
            # Alocando os bmus
            self._bmu = bmus
            print("Criando codebook...")
            self.codebook = Codebook(self.mapsize, self.lattice, self.mapshape)
            self.codebook.matrix = self._normalizer.normalize_by(self.data_raw, trained_neurons.iloc[:,8:].values)
            
            try:
                print("Carregando matriz de distâncias...")
                self._distance_matrix = np.load("Resultados/distance_matrix.npy")
                if self.mapsize[0]*self.mapsize[1] != self._distance_matrix.shape[0]:
                    self._distance_matrix = self.calculate_map_dist
            except:
                self._distance_matrix = self.calculate_map_dist
        else:
            self.data_missing = {"indices":tuple(zip(*np.argwhere(np.isnan(self._data)))), 
                                 "nan_values":None}
            # Para acessar preencher os dados faltantes basta:
            # nan_array[indices] = nan_values
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
    def params_json(self):
        """
        Função de propriedade de classe parar gerar um arquivo csv com os
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
        elif self.missing == False:
            fixed_euclidean_x2 = np.einsum('ij,ij->i', self._data, self._data)
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
        # ESSA PARTE DO CODIGO PODE SER ACELERADA COM PROCESSAMENTO PARALELO
        print("Inicializando mapa...")
        #for i in tqdm(range(nnodes), desc="Matriz\
        #    de distâncias", unit=" Neurons"):
        #    distance_matrix[i] = self.codebook.grid_dist(i).reshape(1, nnodes)
        
        # Acelerar o codigo com processamento paralelo
        def chunk_distmat_fill(nodes):
            for i in tqdm(nodes, desc="Matriz\
                de distâncias", unit=" Neurons"):
                dist = self.codebook.grid_dist(i)
                distance_matrix[i:,i] = dist[i:]
                distance_matrix[i,i:] = dist[i:]
                del dist

        
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
    def bmu_matrix(self):
        """
        Retorna a matriz de BMUs denormalizada. No formato de array com os
        valores dos vetores de cada BMU.
        """
        # Diferenciar a forma de carregar caso se trate de um carregamento de
        # dados treinados
        
        return self._normalizer.denormalize_by(self.data_raw, self.codebook.matrix)

    @property
    def bmu_dataframe(self):
        """
        Função para criar um dataframe dos BMUs resultantes do treinamento. São
        retornados no formado de um DataFrame dos BMUs e suas coordenadas
        retangulares e cúbicas.
        """
        # Criar dataframe
        bmu_df = pd.DataFrame(self.bmu_matrix,
                        index = list(range(1, self.bmu_matrix.shape[0]+1)),
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
        bmu_df.insert(0, "Cub_z", cub_coordinates[:,2])
        bmu_df.insert(0, "Cub_y", cub_coordinates[:,1])
        bmu_df.insert(0, "Cub_x", cub_coordinates[:,0])
        bmu_df.insert(0, "Ret_y", rec_coordinates[:,1])
        bmu_df.insert(0, "Ret_x", rec_coordinates[:,0])
        bmu_df.insert(0, "Udist", min_max_scaler.fit_transform(self.build_umatrix().reshape(-1, 1)))    
        bmu_df.insert(0, "Pind", min_max_scaler.fit_transform(self.build_pmatrix().reshape(-1, 1)))
        bmu_df.insert(0, "BMU", list(range(1, self.bmu_matrix.shape[0]+1)))

        return bmu_df.astype({"BMU": int,
                              "Ret_x": int,
                              "Ret_y": int,
                              "Cub_x": int,
                              "Cub_y": int,
                              "Cub_z": int,
                              "Udist": np.float32,
                              "Pind": np.float32,
                                  })

    @property
    def results_dataframe(self):
        """
        Função para criar um datafreme com os BMUS e seus valores associados a
        cada vetor de entrada.
        """
        # Resgatar o dataframe de bmus
        bmu_df = self.bmu_dataframe
        bmus = self._bmu[0].astype(int)
        #results_array = np.zeros((self._data.shape[0], bmu_df.shape[1]))

        results_df = bmu_df.iloc[bmus,:]
            
        # Preencher iterativamente o df de resultados
        #print("Preenchendo tabela de resultados...")
        #interator = enumerate(bmus)
        #pbar = tqdm(interator, mininterval=1)
        #for i,bmu in pbar:
        #    results_array[i,:] = bmu_df.values[bmu,:]
        
        # Preencher dataframe com as iterações
        #results_df = pd.DataFrame(results_array, columns = bmu_df.columns, index=self._sample_names)

        # Inserir o erro de quantização para cada vetor
        results_df.insert(1, "q-error", self.calculate_quantization_error_expanded)

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
        if self.pred_size>0:
            text_file.write(f"Labeled Variables: Yes\n")
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
    def input_missing(self, save=True):
        """
        Retorna os dados com os valores imputados nas células de entrada vazias.

        Args:
            save: valor booleano para indicar se o arquivo criado será ou não
            salvo dentro do diretório [Imputação].

        Retorna:
            DataFrame com dos dados de entrada com celulas vazias imputadas
                pelos seus respectivos BMUs.

        """
        # Captar os dados
        data = self._data
        data_folder = tempfile.mkdtemp()
        data_name = os.path.join(data_folder, 'data')
        dump(data, data_name)
        data = load(data_name, mmap_mode='r+')
        
        # Preencher com os valores vazios treinados
        data[self.data_missing["indices"]] = self.data_missing["nan_values"]


        # Denormalizar
        data = self._normalizer.denormalize_by(self.data_raw, data)

        # Criar dataframe
        input_df = pd.DataFrame(data, columns = self._component_names, index = self._sample_names)
        if save:
            # Criar diretórios se não existirem
            path = 'Imputação'
            os.makedirs(path, exist_ok=True)
            
            # Salvar
            input_df.to_excel("Imputação/Dados_imputados.xlsx")
            input_df.to_csv("Imputação/Dados_imputados.csv")

        return input_df

    def project_nan_data(self,
                        data_proj,
                        with_labels=True,
                        sample_names=None,
                        save = True):
        """
        Função para projetar dados novos no modelo treinado, mesmo que esses
        dados tenham valores faltantes.

        Args:
            data_proj: Dados se deseja projetar no modelo. Podem estar no
                formato DataFrame ou numpy ndarray.

            with_labels: Valor booleano para indicar se os dados tem as colunas
                de rotulos (modelo de classificação semi-supervisionada) ou não.

        Returns:
            DataFrame com os BMUs representantes de cada vetor de entrada.

        """
        # Checar formatos para adptação
        if isinstance(data_proj, pd.DataFrame):
            sample_names = sample_names if sample_names else data_proj.index
            data_proj = data_proj.values
        elif isinstance(data_proj, np.ndarray):
            data_proj = data_proj
            sample_names = sample_names if sample_names else [f"Amostra_proj_{i}" for i in range(1,data_proj.shape[0]+1)]
        else:
            print("Somente os formatos DataFrame e Ndarray são aceitos como entrada")

        # Checar presença de rótulos de treinamento nos dados a serem projetados
        if with_labels:
            # Retirar as variaveis label dos dados
            data_proj = data_proj[:, :(data_proj.shape[1] - self.pred_size)]
        else:
            data_proj = data_proj

        # Normalizar os valores dos dados
        data_proj = self._normalizer.normalize_by(self.data_raw,
                                                  data_proj,
                                                  with_labels=with_labels,
                                                  pred_size=self.pred_size)

        # Encontrar o BMU para esses novos dados
        bmus = self._find_bmu(data_proj, project=True)
        bmus_ind = bmus[0].astype(int)

        # Resgatar o dataframe de bmus
        bmu_df = self.bmu_dataframe

        # Criar dataframe para projecao
        projected_df = bmu_df.iloc[bmus_ind,:]

        ## Preencher dataframe de projeção com os vetores BMU treinados
        #for i, bmu in enumerate(bmus_ind):
        #    projected_df.iloc[i,:] = bmu_df.iloc[bmu,:]

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
              history_plot = False):
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

        print("Iniciando Treinamento")

        # Aplicar o tipo de inicialização escolhida
        if self.missing:
            if self.load_param:
                self.codebook.pretrain()
            elif self.initialization == 'random':
                self.codebook.random_initialization(self._data)
            elif self.initialization == 'pca':
                print("Ainda não implementado")
        else:
            if self.load_param:
                self.codebook.pretrain()
            elif self.initialization == 'random':
                self.codebook.random_initialization(self._data)
            elif self.initialization == 'pca':
                self.codebook.pca_linear_initialization(self._data)

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
                    self.bmu_dataframe.to_excel(f"Resultados/{self.name}_BMUS.xlsx")
                    self.results_dataframe.to_csv(f"Resultados/{self.name}_resultados.csv")
                    self.bmu_dataframe.to_csv(f"Resultados/{self.name}_BMUS.csv")
                elif dtypes == "xlsx":
                    self.results_dataframe.to_excel(f"Resultados/{self.name}_resultados.xlsx")
                    self.bmu_dataframe.to_excel(f"Resultados/{self.name}_BMUS.xlsx")
                elif dtypes == "csv":
                    self.results_dataframe.to_csv(f"Resultados/{self.name}_resultados.csv")
                    self.bmu_dataframe.to_csv(f"Resultados/{self.name}_BMUS.csv")
                elif dtypes == "parquet":
                    self.results_dataframe.to_parquet(f"Resultados/{self.name}_resultados.parquet")
                    self.bmu_dataframe.to_parquet(f"Resultados/{self.name}_BMUS.parquet")
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
            # self.codebook.nnodes -> numero de nós (nodes mapaN x mapaM)
            # _dlen -> tamanho dos dados (linhas)
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
                data = self._data
                data_folder = tempfile.mkdtemp()
                data_name = os.path.join(data_folder, 'data')
                dump(data, data_name)
                data = load(data_name, mmap_mode='r')

            else:
                data = self._data
                

            #Barra de Treinamento
            pbar = tqdm(range(trainlen), mininterval=1)
            for i in pbar:
                if self.bootstrap:
                    # Passar todos os dados na ultima epoca de treinamento, para garantir que
                    # todos os vetores tenham BMU
                    if self.actual_train == "Fino" and self.train_finetune_len == i:
                        pass
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
                    fixed_euclidean_x2 = np.einsum('ij,ij->i', data[bootstrap_i], data[bootstrap_i])

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
                    fixed_euclidean_x2 = np.einsum('ij,ij->i', data, data)

                    # Atualização da barra de progresso
                    pbar.set_description(f"Época: {i+1}. Raio:{round(radius[i],2)}. QE: {round(np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2)),4)}")
                    
                    # Atualizar os BMUs
                    self._bmu = bmu


        # Processo de treinamento para inputs com dados faltantes
        elif self.missing == True:
            if shared_memory:
                data = self._data
                data_folder = tempfile.mkdtemp()
                data_name = os.path.join(data_folder, 'data')
                dump(data, data_name)
                data = load(data_name, mmap_mode='r+')

            else:
                data = self._data


            #Barra de Progresso
            pbar = tqdm(range(trainlen), mininterval=1)
            for i in pbar:
                if self.bootstrap:
                    print("Ainda precisa otimizar a implementação de treinamento com dados faltantes e bootstrap: lembrar de implementar treinar com dados completos e completar os dados faltantes.")
                    pass
                    # Passar todos os dados na ultima epoca de treinamento, para garantir que
                    # todos os vetores tenham BMU
                    if self.actual_train == "Fino" and self.train_finetune_len == i:
                        data = self._data
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

                    # Apresenta matriz inputada no treinamento fino
                    elif self.actual_train == "Fino":
                        #preenche os valores faltantes no data com os dados da interacao anterior
                        data[self.data_missing["indices"]] = self.data_missing["nan_values"]

                        # Encontra os dados dos BMUs
                        bmu = self._find_bmu(data, njb=njob)


                    #Preencher os locais de dados vazios com os valores encontrados nos bmus
                    # a cada iteração
                    nan_mask = np.isnan(self.data_raw)

                    # Fator de regularização
                    reg = radius[i]/self.total_radius-1/self.total_radius
                    self.reg_hist.append(reg)

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

                    # Adicionar nohistorico de processamento de nan
                    self.nan_value_hist.append(self.data_missing["nan_values"])

                    # Atualização da barra de progresso
                    pbar.set_description(f"Época: {i+1}. Raio:{round(radius[i],2)}. QE: {round(np.mean(bmu[1]),4)}. Reg:{round(reg,2)}")

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

                    # Apresenta matriz inputada no treinamento fino
                    elif self.actual_train == "Fino":
                        #preenche os valores faltantes no data com os dados da interacao anterior
                        data[self.data_missing["indices"]] = self.data_missing["nan_values"]

                        # Encontra os dados dos BMUs
                        bmu = self._find_bmu(data, njb=njob)


                    #Preencher os locais de dados vazios com os valores encontrados nos bmus
                    # a cada iteração
                    nan_mask = np.isnan(self.data_raw)

                    # Fator de regularização
                    reg = radius[i]/self.total_radius-1/self.total_radius
                    self.reg_hist.append(reg)

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

                    # Adicionar nohistorico de processamento de nan
                    self.nan_value_hist.append(self.data_missing["nan_values"])

                    # Atualização da barra de progresso
                    pbar.set_description(f"Época: {i+1}. Raio:{round(radius[i],2)}. QE: {round(np.mean(bmu[1]),4)}. Reg:{round(reg,2)}")

                    if self.history_plot:
                        if i%2 == 0:
                            self.plot_umatrix(figsize = (5,3),
                                         hits = True,
                                         save = True,
                                         file_name = f"{self.actual_train}_epoca{i+1}",
                                         bmu=bmu)
                    # Atualizar bmus
                    self._bmu = bmu



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
            if self.missing:
                # Mapeia os pedaços de dado e aplica o método chunk_bmu_finder em cada pedaço, achando os bmus para cada pedaço
                b = pool.map(lambda chk: chunk_bmu_finder(chk,
                                    self.codebook.matrix[:, :input_matrix.shape[1]],
                                                          nth=nth, 
                                                          missing=True),
                                    chunks)
            else:
                # Mapeia os pedaços de dado e aplica o método chunk_bmu_finder em cada pedaço, achando os bmus para cada pedaço
                b = pool.map(lambda chk: chunk_bmu_finder(chk,
                                    self.codebook.matrix[:, :input_matrix.shape[1]], 
                                                          y2,
                                                          nth=nth),
                             chunks)
                
        else:
            if self.missing:
                # Mapeia os pedaços de dado e aplica o método chunk_bmu_finder em cada pedaço, achando os bmus para cada pedaço
                b = pool.map(lambda chk: chunk_bmu_finder(chk,
                                                          self.codebook.matrix,
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


    def _update_codebook_voronoi(self, training_data, bmu, neighborhood):
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

        # Pega todos os números de bmus de cada linha de dado e coloca no
        # formato Int
        row = bmu[0].astype(int)

        # Todos os índices para as colunas
        col = np.arange(training_data.shape[0])

        # array com 1 repetido no tamanho das linhas de dado
        val = np.tile(1, training_data.shape[0])

        # Cria uma matrix esparsa (csr -> compressed sparsed row) com a chamada
        # csr_matrix((val, (row, col)), [shape=(nnodes, dlen)])
        # Essa chamada indica o formato a[row[k], col[k]] = val[k]
        # Resulta numa matriz [nós, amostras] onde cada nó considerado bmu é
        # marcado como 1
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
    
    @property
    def topographic_error(self):
        """
        Função que está no SOMPY, discordo dessa função, ela busca somente se o primeiro e segundo bmus que 
        melhor representam o vetor de entrada são vizinhos.
        """
        bmus2 = self._find_bmu(self._data, nth=2)
               
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
        if self.missing:
            return np.mean(self._bmu[1])
        else:
            fixed_euclidean_x2 = np.einsum('ij,ij->i', self._data, self._data)
            return np.mean(np.sqrt(self._bmu[1] + fixed_euclidean_x2))
    
    @property
    def calculate_quantization_error_expanded(self):
        if self.missing:
            return self._bmu[1]
        else:
            fixed_euclidean_x2 = np.einsum('ij,ij->i', self._data, self._data)
            return np.sqrt(self._bmu[1] + fixed_euclidean_x2)
        

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
    
    def build_pmatrix(self):
        """
        Função para calcular a matriz P de posição.
        """
        # tamanho do mapa
        Nrows = self.mapsize[1]
        Ncols = self.mapsize[0]
        
        # numeração sequencial iniciando em 0
        sequential_num = np.arange(0,Nrows*Ncols).reshape(Nrows,Ncols)
        
        # Matriz identidade laplaciana de vizinhança
        Laplacian = np.identity(Nrows*Ncols)
        
        # Laplacian - toroidal periodic - marking 1st neighbors of hexagon
        for kk in range(Nrows*Ncols): # percorre todos neuronios
            # position in map
            Ipos = kk // Ncols
            Jpos = kk %  Ncols
            
            # 1st neighbros in clockwise fashion, begins top 12oclock
            # up
            newpos = sequential_num[ (Ipos+1)%Nrows, (Jpos+0)%Ncols ]
            Laplacian[ kk, newpos] = 1
            
            # up right
            if (Ipos % 2) != 0:
                newpos = sequential_num[ (Ipos+1)%Nrows, (Jpos+1)%Ncols ]
                Laplacian[ kk, newpos] = 1
            
            # right
            newpos = sequential_num[ (Ipos+0)%Nrows, (Jpos+1)%Ncols ]
            Laplacian[ kk, newpos] = 1
            
            # low right
            if (Ipos % 2) != 0:
                newpos = sequential_num[ (Ipos-1)%Nrows, (Jpos+1)%Ncols ]
                Laplacian[ kk, newpos] = 1
                
            # low
            newpos = sequential_num[ (Ipos-1)%Nrows, (Jpos+0)%Ncols ]
            Laplacian[ kk, newpos] = 1
            
            # low left
            if (Ipos % 2) == 0:
                newpos = sequential_num[ (Ipos-1)%Nrows, (Jpos-1)%Ncols ]
                Laplacian[ kk, newpos] = 1
            
            # left
            newpos = sequential_num[ (Ipos+0)%Nrows, (Jpos-1)%Ncols ]
            Laplacian[ kk, newpos] = 1
            
            # up left
            if (Ipos % 2) == 0:
                newpos = sequential_num[ (Ipos+1)%Nrows, (Jpos-1)%Ncols ]
                Laplacian[ kk, newpos] = 1
            
        # Cuthill opera em.matriz esparsa, então converter para csr
        Laplacian_sp = sp.csr_matrix(Laplacian)

        # cuthill
        remapping = sp.csgraph.reverse_cuthill_mckee(Laplacian_sp, symmetric_mode=False)
        
        # renumera o mapa
        reorder = remapping.copy()

        for ii in range(Nrows*Ncols):
            reorder[ii] = remapping.tolist().index(ii)

        # mapa nova numeração 
        renumbered_map = reorder.reshape(Nrows, Ncols)
        
        return renumbered_map

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

    def _chunk_based_bmu_find(self,
                            input_matrix, 
                            codebook, 
                            y2=None, 
                            nth=1, 
                            missing=False):
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

                # Calculo das distâncias usando sklearn nan_euclidean_distances
                # (tem um fator interno de peso para valores faltantes)
                d = nan_euclidean_distances(codebook, ddata)

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

                # Calculo de distancias usando operacoes matriciais
                d = np.dot(codebook, ddata.T)
                d *= -2
                d += y2.reshape(nnodes, 1)

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