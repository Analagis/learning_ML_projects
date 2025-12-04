import re
import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Lasso
import seaborn as sns
from IPython.display import display


def column_clearing(column):
    return column.apply(lambda x: re.sub(r'[ \[\]\'\"]', '', str(x)))
    

def columns_creating(features, df):
    for feature in features:
        df[feature] = df['features'].apply(lambda x: 1 if feature in x else 0)

class ClusterLassoPipeline:
    def __init__(self, dataframe, features_list, target, cluster_features, clustering_methods, n_clusters=30, test_size=0.2, random_state=42):
        self.df = dataframe.copy()
        self.features_list = features_list
        self.target = target
        self.cluster_features = cluster_features
        self.clustering_methods = clustering_methods
        self.n_clusters = n_clusters
        self.test_size = test_size
        self.random_state = random_state
        
        # Separate X: features (excluding target and cluster_features)
        self.X_features = [f for f in self.features_list if f != self.target and f not in self.cluster_features]
        
        # Split datasets later
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_cluster_train, self.X_cluster_test = self.split()

        # Storage for results
        self.cluster_labels_train = {}
        self.cluster_labels_test = {}
        self.silhouette_scores = {}
        self.distortions = {}
        self.fit_times = {}
        self.feature_importances = {}
        self.regression_metrics = {}
        
        # For plots data (cluster labels for train)
        self.plot_cluster_labels = {}
   
    def split(self):
        # Prepare X, y and train/test split
        X_all = self.df[self.X_features].copy()
        y_all = self.df[self.target].copy()
        X_cluster_all = self.df[self.cluster_features].copy()
        
        return train_test_split(X_all, y_all, X_cluster_all, test_size=self.test_size)
        #self.X_cluster_train, self.X_cluster_test = X_cluster_all.iloc[:int(len(X_cluster_all)*(1-self.test_size))], X_cluster_all.iloc[:int(len(X_cluster_all)*(1-self.test_size))]


    def fit(self):
        # Baseline Lasso without clusters for comparison
        self._run_lasso()
        self.silhouette_scores['baseline'] = {'train': np.nan, 'test': np.nan}
        self.distortions['baseline'] = {'train': np.nan, 'test': np.nan}
        self.fit_times['baseline'] = np.nan
            
        for method_name, clusterer in self.clustering_methods.items():
            start_time = time.time()
            print(f"Обучение {method_name}")

            # fit/predict для каждого кластеризатора с учетом типа
            if hasattr(clusterer, 'fit_predict'):  
                # Для алгоритмов с fit_predict напрямую (например DBSCAN, AgglomerativeClustering)
                labels_train = clusterer.fit_predict(self.X_cluster_train)
                try:
                    labels_test = clusterer.fit_predict(self.X_cluster_test)
                except:
                    # Если нельзя fit_predict на test (например DBSCAN), то метки -1
                    labels_test = np.array([-1]*len(self.X_cluster_test))
            elif hasattr(clusterer, 'fit') and hasattr(clusterer, 'predict'):
                clusterer.fit(self.X_cluster_train)
                labels_train = clusterer.predict(self.X_cluster_train)
                labels_test = clusterer.predict(self.X_cluster_test)
            else:
                raise ValueError(f"Clustering estimator {method_name} missing fit/predict methods")
            
            fit_time = time.time() - start_time
            
            # Calculate silhouette scores (only if more than 1 cluster present)
            try:
                sil_train = silhouette_score(self.X_cluster_train, labels_train) if len(set(labels_train)) > 1 else np.nan
            except:
                sil_train = np.nan
            try:
                sil_test = silhouette_score(self.X_cluster_test, labels_test) if len(set(labels_test)) > 1 else np.nan
            except:
                sil_test = np.nan
            
            # Calculate distortion
            distortion_train = np.nan
            distortion_test = np.nan
            if 'kmeans' in method_name:
                distortion_train = clusterer.inertia_
                # For test set distortion: compute sum of distances to nearest cluster center
                centers = clusterer.cluster_centers_
                dist_test = np.min(np.linalg.norm(self.X_cluster_test.values[:, None] - centers[None, :], axis=2), axis=1)
                distortion_test = dist_test.sum()
            elif 'gmm' in method_name:
                # use negative log-likelihood as distortion proxy
                distortion_train = -clusterer.score(self.X_cluster_train) * len(self.X_cluster_train)
                distortion_test = -clusterer.score(self.X_cluster_test) * len(self.X_cluster_test)
            
            # Save results
            self.silhouette_scores[method_name] = {'train': sil_train, 'test': sil_test}
            self.distortions[method_name] = {'train': distortion_train, 'test': distortion_test}
            self.fit_times[method_name] = fit_time

            # Store labels for plotting
            self.plot_cluster_labels[method_name] = labels_train

            # Add cluster labels to X_train and X_test
            self.X_train['cluster'] = labels_train
            self.X_test['cluster'] = labels_test
            
            self._run_lasso(method_name)
            
    def _run_lasso(self, method_name = 'baseline'):
        """Baseline Lasso without cluster features for comparison."""
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        lasso = Lasso(random_state=self.random_state)
        lasso.fit(X_train_scaled, self.y_train)
        y_pred_train = lasso.predict(X_train_scaled)
        y_pred_test = lasso.predict(X_test_scaled)
        
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
        rmse_train = root_mean_squared_error(self.y_train, y_pred_train)
        rmse_test = root_mean_squared_error(self.y_test, y_pred_test)
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(self.y_test, y_pred_test)
        
        importance = pd.Series(abs(lasso.coef_), index=self.X_train.columns)
        importance_sorted = importance.sort_values(ascending=False)

        self.feature_importances[method_name] = importance_sorted
        
        self.regression_metrics[method_name] = {
            'MAE': {'train': mae_train, 'test': mae_test},
            'RMSE': {'train': rmse_train, 'test': rmse_test},
            'R2': {'train': r2_train, 'test': r2_test}
        }
            
    def plot_results(self):
        n_methods = len(self.clustering_methods)
        methods = self.clustering_methods
        
        # Determine subplot layout (4 per row max)
        ncols = min(4, n_methods)
        nrows = (n_methods + ncols - 1) // ncols
        
        fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
        plot_vars = self.cluster_features
        
        is_3d = len(plot_vars) == 3
        for i, method in enumerate(methods):
            ax = fig.add_subplot(nrows, ncols, i+1, projection='3d' if is_3d else None)
            
            labels = self.plot_cluster_labels.get(method)
            if labels is None:
                continue
            if is_3d:
                ax.scatter(self.X_cluster_train.iloc[:,0], self.X_cluster_train.iloc[:,1], self.X_cluster_train.iloc[:,2], c=labels, cmap='tab20', s=10)
                ax.set_xlabel(plot_vars[0])
                ax.set_ylabel(plot_vars[1])
                ax.set_zlabel(plot_vars[2])
            else:
                ax.set(xlim=(-74.05,-73.85))
                ax.set(ylim=(40.65,40.85))
                plot_df = self.X_cluster_train.copy()
                plot_df['cluster'] = self.plot_cluster_labels[method]
                sns.scatterplot(
                data=plot_df,
                x=plot_vars[0],
                y=plot_vars[1],
                hue='cluster',
                ax=ax,
                s=10,
                legend=True,
                linewidth=0,
                alpha=0.6  # Можно добавить прозрачность
            )
            
            ax.set_title(f'Clusters by {method}')
        
        plt.tight_layout()
        plt.show()
        
    def feature_importances_table(self):
        # Combine all feature importances into DataFrame, align by index filling NaNs when missing
        importances_df = pd.DataFrame(self.feature_importances)
        display(importances_df.sort_values(by = "baseline", ascending=False))
    
    def metrics_table(self):
        # Combine Silhouette, Distortion, Time for train/test in multiindex DataFrame
        rows = []
        for metric_name in ['silhouette', 'distortion']:
            for phase in ['train', 'test']:
                row = {}
                for method in self.silhouette_scores.keys():
                    if metric_name == 'silhouette':
                        val = self.silhouette_scores[method][phase]
                    else:
                        val = self.distortions[method][phase]
                    row[method] = val
                rows.append((metric_name, phase, row))
        
        # Add times (only one per method, for training)
        row_time = {method: self.fit_times[method] for method in self.fit_times.keys()}
        rows.append(('fit_time', '-', row_time))
        
        # Build MultiIndex
        index = pd.MultiIndex.from_tuples([(m,p) for m,p,_ in rows], names=['Metric', 'Phase'])
        data = [r for _,_,r in rows]
        df = pd.DataFrame(data, index=index)
        display(df)
    
    def regression_metrics_table(self):
        # Combine MAE, RMSE, R2 for train/test in multiindex DataFrame
        metrics = ['MAE', 'RMSE', 'R2']
        phases = ['train', 'test']
        rows = []
        for metric in metrics:
            for phase in phases:
                row = {}
                for method in self.regression_metrics.keys():
                    row[method] = self.regression_metrics[method][metric][phase]
                rows.append((metric, phase, row))
        index = pd.MultiIndex.from_tuples([(m,p) for m,p,_ in rows], names=['Metric', 'Phase'])
        data = [r for _,_,r in rows]
        df = pd.DataFrame(data, index=index)
        display(df)
