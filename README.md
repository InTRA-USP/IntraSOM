# IntraSOM
-----
Intrasom é uma implementação completamente baseada em Python de mapas auto-organizáveis (SOMs) desenvolvido 
pelo centro de pesquisas Integrated Technology for Rock and Fluid Analysis (InTRA) (https://www.usp.br/intra/). 
IntraSOM é construído usando Programação Orientada a Objetos e inclui suporte para grades hexagonais, 
topologias toroidais e uma ampla gama de ferramentas de visualização para melhorar a análise, exploração 
e classificação de conjuntos de dados complexos. Além disso, IntraSOM inclui recursos para lidar com 
dados ausentes durante o treinamento e para algoritmos eficientes de agrupamento. O artigo tem como 
objetivo tornar os SOMs mais acessíveis a pesquisadores e profissionais em diversos campos, fornecendo 
uma implementação Python abrangente de SOMs e um framework para expandir e implementar facilmente outros 
algoritmos baseados em SOM.

A estrutura dessa biblioteca é baseada na estrutrua da biblioteca SOMPY de Moosavi et al (2014). Com implementação de:
* Treinamento em topologia toroidal
* Treinamento em célula hexagonal
* Treinamento com dados faltantes
* Imputação de dados
* Carregamento de um treinamento performado anteriormente.
* Módulo de avaliação do treinamento semi-supervisionado com plotagem de curva ROC.
* Módulo de Plotagem e calculo de matriz U e mapa de componentes de treinamento.
* Salvamento dos dados treinamento.
* Geração de Relatório de Treinamento.
* Projeção de novos dados em um mapa treinado.
* Módulo de agrupamento dos neurônios treinados com k-means e vizualização dos resultados.

Changes
* Aceleração da função de criação da matriz de distâncias, dobro da velocidade.
* Utilizar operações matriciais para o encontro dos bmus 
no treinamento de dados completos.
* Formula de quantização do erro para as operações matriciais de treinamento de dados
completos.
* Formato .parquet implementado para saida de dados.
* Plotagem de rótulos na Matriz U

Dependencies:
As dependências do pacote IntraSOM são:
- numpy
- scipy
- scikit-learn
- pandas
- tqdm
- scipy
- matplotlib 

Installation:
Ainda não upado para PyPl.

Citation

https://github.com/InTRA-USP/IntraSOM

# IntraSOM
