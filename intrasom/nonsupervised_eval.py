from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from typing import Union

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

    def knn_oversampling(
        self,
        samples_labels: Union[list, np.ndarray, pd.Series],
        pca_plot: bool = True,
        plt_figsize: tuple = (10,5),
        plt_title: str = 'SOM and KNN oversampling',
        normalized: bool = True,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        algorithm: str = 'auto',
        leaf_size: int = 30,
        p: int = 2,
        metric: str = 'minkowski',
        metric_params = None,
        n_jobs: int = -1):

        """
        Function to perform KNN oversampling of the data based on the neurons.
        Each BMU is used as a candidate for new samples and the KNN analysis.
        The number of samples generated will vary since not every neuron is a BMU.

        Args:
            samples_labels: Labels of the samples, in the same order as in the training dataset.

            pca_plot: If True, a PCA plot will be generated showing the original samples and the new samples (BMUs).

            normalized: If True, the data will be normalized (useful when features have different scales).
                If False, the raw data will be used (it can be more fair if features have the same scales). Default is True.

            n_neighbors: Number of neighbors to use by default for k_neighbors queries.

            weights: Weight function used in prediction. Possible values:
                'uniform' : uniform weights. All points in each neighborhood are weighted equally.
                'distance' : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
                [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.

            algorithm: Algorithm used to compute the nearest neighbors:
                'ball_tree' will use BallTree
                'kd_tree' will use KDTree
                'brute' will use a brute-force search.
                'auto' will attempt to decide the most appropriate algorithm based on the values passed to fit method.

            leaf_size: Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.

            p: Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

            metric: the distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.

            metric_params: Additional keyword arguments for the metric function.

            n_jobs: The number of parallel jobs to run for neighbors search. -1 means using all processors.

        Returns:
            A dataframe with the new samples, the index is the neuron number and the column 'Label' contains the predicted label.
            """

        samples = self.som._data if normalized else self.som.data_raw

        le = LabelEncoder()
        samples_labels = le.fit_transform(samples_labels) # convert to int labels

        bmu_indices = self.som.results_dataframe['Neuron'].unique() - 1 # -1 to convert to python index
        bmus = self.som.codebook.matrix[bmu_indices,:] if normalized else self.som.neurons_dataframe.iloc[bmu_indices,7:].values

        # Find the labels for the BMUs
        neigh = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs)
        neigh.fit(samples, samples_labels)
        bmu_labels = neigh.predict(bmus)

        # Create the new samples dataframe
        new_samples = pd.DataFrame(data=bmus, columns=self.som.component_names)
        new_samples['Label'] = le.inverse_transform(bmu_labels) # convert the int labels to the original labels
        # Add the Neuron number as the index of the dataframe
        new_samples.index = bmu_indices + 1 # +1 to convert to neuron numbering

        print(f'SOM and KNN oversampling result')
        print(f'Mapsize: {self.som.mapsize[0]}x{self.som.mapsize[1]} ({self.som.mapsize[0]*self.som.mapsize[1]} neurons)')
        print(f'Number of new samples (BMUs): {len(bmus)}')

        if pca_plot:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(samples)
            pca_bmu = pca.transform(bmus)

            fig, ax = plt.subplots(figsize=plt_figsize)
            scatter_og = sns.scatterplot(
                x=pca_data[:,0], 
                y=pca_data[:,1], 
                hue=le.inverse_transform(samples_labels),
                hue_order=le.classes_,
                palette='gist_rainbow', 
                alpha=0.3, 
                edgecolor='k', 
                legend=False,
                ax=ax
            )
            scatter_new = sns.scatterplot(
                x=pca_bmu[:,0], 
                y=pca_bmu[:,1], 
                hue=new_samples["Label"],
                hue_order=le.classes_,
                palette='gist_rainbow', 
                marker='h',
                edgecolor='k', 
                s=80,
                legend='full',
                ax=ax
            )

            # Get the handles/labels from the scatterplot of new samples
            handles_color, labels_color = scatter_new.get_legend_handles_labels()
            legend1 = ax.legend(
                handles_color, labels_color,
                title_fontsize='12',
            )
            ax.add_artist(legend1) 

            # Create custom legend
            og_marker = mlines.Line2D(
                [], [], 
                color='gray', 
                marker='o', 
                linestyle='None', 
                markersize=6,
                markeredgecolor='k',
                alpha=0.3, # Match the alpha used in the scatterplot
                label='Original samples'
            )
            new_marker = mlines.Line2D(
                [], [], 
                color='gray',
                marker='h', 
                linestyle='None', 
                markersize=9,
                markeredgecolor='k',
                label='New samples (BMUs)'
            )

            ax.legend(handles=[og_marker, new_marker], title='', loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
            # Adjust the plot to make room for the legend
            plt.subplots_adjust(bottom=0.2)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title(plt_title)
            plt.show()

        return new_samples