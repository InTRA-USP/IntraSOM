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
        Returns the semi-supervised learning evaluation dataframe.
        Args:
            data_test: Test data, in dataframe or numpy ndarray format.            

            bayesian_thresh: Need to implement. Allows you to perform Bayesian evaluation of
                thresholds.
        """
        
        # Check formats for adaptation
        if isinstance(data_test, pd.DataFrame):
            sample_names = data_test.index.values
            labels = [label for label in data_test.iloc[:,-self.pred_size:].columns]
            data_test = data_test.values
        elif isinstance(data_test, np.ndarray):
            data_test = data_test
            sample_names = [f"Test_sample_{i}" for i in range(1,data_test.shape[0]+1)]
            labels = [f"Var{i}" for i in range(self.pred_size)]
        else:
            print("Only DataFrame and ndarray formats are accepted as input")
        
        # True labels
        y_true = data_test[:,-self.pred_size:]
        # Project values
        y_pred = self.som.project_nan_data(data_test, 
                                           with_labels=True, 
                                           save=False, 
                                           sample_names = sample_names).iloc[:,-self.pred_size:].values
        
        
        
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
            print("Saving...")
            path = 'Evaluation'
            os.makedirs(path, exist_ok=True)
            if best_lim:
                report_df.to_excel(f"Evaluation/Evaluation_SOM_best_lim{self.name}.xlsx")
            else:
                report_df.to_excel(f"Evaluation/Evaluation_SOM_{self.name}.xlsx")
        if plot_roc:
            for i, label in enumerate(labels):
                if report_df["Total Positives"].values[i]!=0:
                    fper, tper, _ = roc_curve(y_true[:,i], y_pred[:,i])
                    roc_score = roc_auc_score(y_true[:,i], y_pred[:,i])
                    plt.figure(figsize=(7,7))
                    plt.plot(fper, tper, color='red', label='ROC')
                    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f"ROC {label[:10]}")
                    plt.text(0.4, 0.2, f"AUC:{round(roc_score,2)}", fontsize=12)
                    plt.legend()
                    if save_roc:
                        path = 'Evaluation/ROC'
                        os.makedirs(path, exist_ok=True)
                        plt.savefig(f"Evaluation/ROC/ROC_{label[:7]}.png", dpi=200)
                    plt.show()
        
        return report_df
    
    def evaluate(self, y_pred, y_true, thresh, sample_names):
        """
        Function to evaluate a semi-supervised training. The evaluation metrics are:
        [ADD A SHORT DEFINITION OF EACH OF THEM]
        * True Negatives (TN)
        * False Negatives (FN)
        * True Positives (TP)
        * False Positives (FP)
        * Accuracy
        * Total Positives
        * Sensitivity
        * Specificity
        * Accuracy
        * False Positive Rate
        * False Negative Rate

        Args:
            y_pred = label values ​​predicted by the SOM model.

            y_true = True label values ​​from the test set.

            thresh = threshold for identifying labels as positive or negative.
                Accepts number format or a list, if you want to apply a
                list of different thresholds for each classifier.


            sample_names = classifiers names, in the same order as in training
                and testing sets.

        Returns:
            A dataframe with the classifiers in the indexes and the training metrics
            in the columns.
        """

        def predict_label(y_pred, thresh):
            """
            Function to predict training labels from a threshold.
            """
            return np.where(y_pred > thresh, 1, 0)
        def divide_nonzero(num, den):
            """
            Function to divide values ​​by 0. The division value is 0.
            """
            return np.divide(num, 
                             den, 
                             out=np.zeros(num.shape, dtype=float), 
                             where=den!=0)

        if len([thresh]) == 1:
            # Predict the labels according to the threshold
            y_pred_thresh = predict_label(y_pred, thresh)
        elif len([thresh]) > 1:
            y_pred_thresh = np.zeros(y_pred.shape)
            for i, ts in enumerate(thresh):
                y_pred_thresh[i] = predict_label(y_pred[i], ts)

        # Create confusion matrix for each label
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
        # Create DataFrame
        evaluation = pd.DataFrame(index=sample_names)

        # Preencher DataFrame
        evaluation["True Negatives"] = TN
        evaluation["False Negatives"] = FN
        evaluation["True Positives"] = TP
        evaluation["False Positives"] = FP
        evaluation["Total Positives"] = PTV
        evaluation["Sensitivity"] = TPR
        evaluation["Specificity"] = TNR
        evaluation["Accuracy"] = PPV
        evaluation["Recall"] = REC
        evaluation["False Positive Rate"] = FPR
        evaluation["False Negative Rate"] = FNR
        evaluation["Accuracy"] = ACC


        return evaluation
    
    def evaluate_thresh(self, data_test, labels=None, plot=False, save=False):
        """
        Function to evaluate variation in accuracy, rate of false negatives and rate of
        false positives along the range of thresholds.
        """
        def predict_label(y_pred, thresh):
            """
            Function to predict training labels from a threshold.
            """
            return np.where(y_pred > thresh, 1, 0)
        def divide_nonzero(num, den):
            """
            Function to divide values ​​by 0. The division value is 0.
            """
            return np.divide(num, 
                             den, 
                             out=np.zeros(num.shape, dtype=float), 
                             where=den!=0)
        
        # Check formats for adaptation
        if isinstance(data_test, pd.DataFrame):
            sample_names = data_test.index
            data_test = data_test.values
        elif isinstance(data_test, np.ndarray):
            data_test = data_test
            sample_names = [f"Test_sample_{i}" for i in range(1,data_test.shape[0]+1)]
        else:
            print("Only DataFrame and ndarray formats are accepted as input")
        
        # True labels
        y_true = data_test[:,-self.pred_size:]
        # Project values
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
                #Plot
                plt.figure(figsize=(8,3))
                plt.plot( np.arange(0,1.001,0.01), ACC, label="Accuracy")
                plt.plot(np.arange(0,1.001,0.01), FNR, label="False Negative Rate")
                plt.plot(np.arange(0,1.001,0.01), FPR, label="False Positive Rate")
                plt.vlines(best_lim[i], 0,100, linestyles ="solid", colors ="k")
                plt.title(f"Threshold Evaluation: {labels[label_index]}")
                plt.xlabel("Thresholds")
                plt.ylabel("Rates")
                #plt.xlim(0,0.11)
                plt.legend()
                if save:
                    path = 'Evaluation/best_lim'
                    os.makedirs(path, exist_ok=True)
                    plt.savefig(f"Evaluation/best_lim/best_lim_{labels[label_index][:7]}.png", dpi=200)
            
        return best_lim
            