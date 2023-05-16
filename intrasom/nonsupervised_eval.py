from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt


class Evaluation(object):
    
    def __init__(self, som_object):
        self.name = som_object.name
        self.pred_size = som_object.pred_size
        self.som = som_object
    
    def evaluation_report(self, 
                          data_test, 
                          labels = None, 
                          bayesian_thresh=False,
                          best_lim = False,
                          save=True, 
                          plot_roc = False,
                          save_roc=False):
        """
        Retorna o dataframe de avaliação do aprendizado semi-supervisionado.
        Args:
            data_test: Dados de teste, no formato dataframe ou numpy ndarray.
            
            bayesian_thresh: Falta implementar. Permite fazer avaliação bayesiana dos 
                limiares.
        """
        
        # Checar formatos para adaptação
        if isinstance(data_test, pd.DataFrame):
            sample_names = data_test.index
            labels = [label for label in data_test.iloc[:,-self.pred_size:].columns]
            data_test = data_test.values
        elif isinstance(data_test, np.ndarray):
            data_test = data_test
            sample_names = [f"Amostra_teste_{i}" for i in range(1,data_test.shape[0]+1)]
            labels = [f"Var{i}" for i in range(self.pred_size)]
        else:
            print("Somente os formatos DataFrame e Ndarray são aceitos como entrada")
        
        # Rótulos verdadeiros
        y_true = data_test[:,-self.pred_size:]
        # Projetar valores
        y_pred = self.som.project_nan_data(data_test, with_labels=True, save=False).iloc[:,-self.pred_size:].values
        
        
        
        if best_lim:
            thresh = self.evaluate_thresh(data_test=data_test, 
                                          labels=labels, 
                                          plot=False, 
                                          save=False)
        else:    
            thresh = 0.5 
        
        report_df = self.evaluate(y_pred, y_true, thresh, labels)
        if best_lim:
            report_df["Thresh"] = thresh
        else:
            report_df["Thresh"] = np.array([thresh]*y_true.shape[1])
        
        if save:
            print("Salvando...")
            path = 'Avaliacao'
            os.makedirs(path, exist_ok=True)
            if best_lim:
                report_df.to_excel(f"Avaliacao/Avaliação_SOM_best_lim{self.name}.xlsx")
            else:
                report_df.to_excel(f"Avaliacao/Avaliação_SOM_{self.name}.xlsx")
        if plot_roc:
            for i, label in enumerate(labels):
                if report_df["Total de Positivos"].values[i]!=0:
                    fper, tper, _ = roc_curve(y_true[:,i], y_pred[:,i])
                    roc_score = roc_auc_score(y_true[:,i], y_pred[:,i])
                    plt.figure(figsize=(7,7))
                    plt.plot(fper, tper, color='red', label='ROC')
                    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
                    plt.xlabel('Taxa de Falsos Positivos')
                    plt.ylabel('Taxa de Verdadeiros Positivos')
                    plt.title(f"ROC {label[:10]}")
                    plt.text(0.4, 0.2, f"AUC:{round(roc_score,2)}", fontsize=12)
                    plt.legend()
                    if save_roc:
                        path = 'Avaliacao/ROC'
                        os.makedirs(path, exist_ok=True)
                        plt.savefig(f"Avaliacao/ROC/ROC_{label[:7]}.png", dpi=200)
                    plt.show()
        
        return report_df
    
    def evaluate(self, y_pred, y_true, thresh, sample_names):
        """
        Função para avaliar um treinamento semi-supervisionado. As métricas de avaliação são:
        [COLOCAR UMA DEFINIÇÃO CURTA DE CADA UM DELES]
        * Verdadeiros Negativos
        * Falsos Negativos
        * Verdadeiros Positivos
        * Falsos Positivos
        * Acurácia
        * Total de Positivos
        * Sensibilidade
        * Especificidade
        * Precisão
        * Taxa de Falsos Positivos
        * Taxa de Falsos Negativos

        Args:
            y_pred = valores dos rótulos preditos pelo modelo SOM.

            y_true = valores de rótulos verdadeiros, do conjunto de teste.

            thresh = limiar para identificação dos rótulos como positivos ou negativos.
                Aceita o formato de número ou uma lista, caso se queira aplicar uma 
                lista de limiares diferentes para cada classificador.
        

            sample_names = nomes dos classificadores, na mesma ordem que nos conjuntos de 
                treinamento e teste.

        Retorna:
            Um dataframe com os classificadores nos índices e as métricas de treinamento
            nas colunas.
        """

        def predict_label(y_pred, thresh):
            """
            Função para predizer labels de treinamento a partir de um limiar.
            """
            return np.where(y_pred > thresh, 1, 0)
        def divide_nonzero(num, den):
            """
            Função para dividir valores por 0. O valor da divisão é 0.
            """
            return np.divide(num, 
                             den, 
                             out=np.zeros(num.shape, dtype=float), 
                             where=den!=0)

        if len([thresh]) == 1:
            # Predizer os labels de acordo com o threshold
            y_pred_thresh = predict_label(y_pred, thresh)
        elif len([thresh]) > 1:
            y_pred_thresh = np.zeros(y_pred.shape)
            for i, ts in enumerate(thresh):
                y_pred_thresh[i] = predict_label(y_pred[i], ts)

        # Criar matriz de confusão para cada label
        cm = multilabel_confusion_matrix(y_true, y_pred_thresh)

        # Separar TN, FN, TP e FP
        TN = np.array([aval[0][0] for aval in cm])
        FN = np.array([aval[1][0] for aval in cm])
        TP = np.array([aval[1][1] for aval in cm])
        FP = np.array([aval[0][1] for aval in cm])

        # Positives on the true values
        PTV = y_true.sum(axis=0).astype(int)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = np.around(divide_nonzero(TP, (TP+FN))*100,2)
        # Specificity or true negative rate
        TNR = np.around(divide_nonzero(TN, (TN+FP))*100,2)
        # Precision or positive predictive value
        PPV = np.around(divide_nonzero(TP, (TP+FP))*100,2)
        # Fall out or false positive rate
        FPR = np.around(divide_nonzero(FP, (FP+TN))*100,2)
        # False negative rate
        FNR = np.around(divide_nonzero(FN, (TP+FN))*100,2)
        # Overall accuracy
        ACC = np.around((TP+TN)/(TP+FP+FN+TN)*100,2)
        # Recall
        REC = np.around(divide_nonzero(TP, PTV)*100,2)
        # Criar DataFrame
        evaluation = pd.DataFrame(index=sample_names)

        # Preencher DataFrame
        evaluation["Verdadeiros Negativos"] = TN
        evaluation["Falsos Negativos"] = FN
        evaluation["Verdadeiros Positivos"] = TP
        evaluation["Falsos Positivos"] = FP
        evaluation["Total de Positivos"] = PTV
        evaluation["Sensibilidade"] = TPR
        evaluation["Especificidade"] = TNR
        evaluation["Precisão"] = PPV
        evaluation["Recall"] = REC
        evaluation["Taxa de Falsos Positivos"] = FPR
        evaluation["Taxa de Falsos Negativos"] = FNR
        evaluation["Acurácia"] = ACC


        return evaluation
    
    def evaluate_thresh(self, data_test, labels=None, plot=False, save=False):
        """
        Função para avaliar variação da acuração, taxa de falsos negativos e taxa de 
        falsos positivos ao longo da variação de thresholds.
        """
        def predict_label(y_pred, thresh):
            """
            Função para predizer labels de treinamento a partir de um limiar.
            """
            return np.where(y_pred > thresh, 1, 0)
        def divide_nonzero(num, den):
            """
            Função para dividir valores por 0. O valor da divisão é 0.
            """
            return np.divide(num, 
                             den, 
                             out=np.zeros(num.shape, dtype=float), 
                             where=den!=0)
        
        # Checar formatos para adptação
        if isinstance(data_test, pd.DataFrame):
            sample_names = data_test.index
            data_test = data_test.values
        elif isinstance(data_test, np.ndarray):
            data_test = data_test
            sample_names = [f"Amostra_teste_{i}" for i in range(1,data_test.shape[0]+1)]
        else:
            print("Somente os formatos DataFrame e Ndarray são aceitos como entrada")
        
        # Rótulos verdadeiros
        y_true = data_test[:,-self.pred_size:]
        # Projetar valores
        y_pred = self.som.project_nan_data(data_test, with_labels=True, save=False).iloc[:,-self.pred_size:].values
        

        best_lim = np.zeros(self.pred_size)
        
        for i in range(self.pred_size):
            ACC = np.zeros(len(np.arange(0,1.0001,0.01)))
            FNR = np.zeros(len(np.arange(0,1.0001,0.01)))
            FPR = np.zeros(len(np.arange(0,1.0001,0.01)))
            
            label_index = i
            
            for j,ts in enumerate(np.arange(0,1.0001,0.01)):
                y_pred_thresh = predict_label(y_pred, ts)

                # Criar matriz de confusão para cada label
                cm = multilabel_confusion_matrix(y_true, y_pred_thresh)

                # Separar TN, FN, TP e FP
                TN = np.array([aval[0][0] for aval in cm])
                FN = np.array([aval[1][0] for aval in cm])
                TP = np.array([aval[1][1] for aval in cm])
                FP = np.array([aval[0][1] for aval in cm])

                # Fall out or false positive rate
                FPR[j] = np.around(divide_nonzero(FP, (FP+TN))*100,2)[label_index]
                # False negative rate
                FNR[j] = np.around(divide_nonzero(FN, (TP+FN))*100,2)[label_index]
                # Overall accuracy
                ACC[j] = np.around((TP+TN)/(TP+FP+FN+TN)*100,2)[label_index]
                
            dist = np.zeros(ACC.shape[0])
            for k, (axx, fnr, fpr) in enumerate(zip(ACC, FNR, FPR)):
                dist[k] = abs(fnr-fpr)
                
            best_lim[i] = 0.01 if np.arange(0,1.001,0.01)[np.argmin(dist)] == 0 else np.arange(0,1.001,0.01)[np.argmin(dist)]
            
            if plot:
                #Plotar
                plt.figure(figsize=(8,3))
                plt.plot( np.arange(0,1.001,0.01), ACC, label="Acurácia")
                plt.plot(np.arange(0,1.001,0.01), FNR, label="Taxa de Falsos Negativos")
                plt.plot(np.arange(0,1.001,0.01), FPR, label="Taxa de Falsos Positivos")
                plt.vlines(best_lim[i], 0,100, linestyles ="solid", colors ="k")
                plt.title(f"Avaliação Limiar: {labels[label_index]}")
                plt.xlabel("Limiares")
                plt.ylabel("Taxas")
                #plt.xlim(0,0.11)
                plt.legend()
                if save:
                    path = 'Avaliacao/best_lim'
                    os.makedirs(path, exist_ok=True)
                    plt.savefig(f"Avaliacao/best_lim/best_lim_{labels[label_index][:7]}.png", dpi=200)
            
        return best_lim
     
            