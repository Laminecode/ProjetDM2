import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances



class ClusteringApp:
    def __init__(self):
        self.data = None
        self.features = None
        self.results = {}
        
    def load_data(self, filepath, numeric_only=False):
        try:
            self.raw_data = pd.read_csv(filepath)
            self.label_encoders = {}
            if numeric_only:
                # Keep only numeric columns
                self.data = self.raw_data.select_dtypes(include=['number'])
            else:
                # Create a copy to avoid modifying raw data
                self.data = self.raw_data.copy()
                for col in self.raw_data.select_dtypes(exclude=['number']).columns:
                    self.data[col] = self.data[col].astype(str).str.lower().str.strip()
                    
                # Process each non-numeric column
                for col in self.raw_data.select_dtypes(exclude=['number']).columns:
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col])
                    self.label_encoders[col] = le
                    print(f"Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
            
            # Drop rows with any missing values
            self.data = self.data.dropna()
            
            # Check if we still have data
            if len(self.data) == 0:
                raise ValueError("No data remaining after preprocessing")
            
            # Store feature names
            self.features = self.data.columns.tolist()
            
            # Initialize and fit scaler
            self.scaler = StandardScaler()
            self.normalized_data = self.scaler.fit_transform(self.data)
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self):
        
        pass
    
    def run_kmeans(self, n_clusters=3, random_state=42, visualize=True):
        """
        Parameters:
        -----------
        n_clusters : int (default=3)
            Number of clusters to form
        random_state : int (default=42)
            Random seed for reproducibility
        visualize : bool (default=True)
            Whether to plot the clusters (using PCA for 2D projection)
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'model': The trained KMeans model
            - 'labels': Cluster assignments
            - 'metrics': Evaluation metrics
        """
        try:
            # Verify data is loaded and preprocessed
            if self.normalized_data is None:
                raise ValueError("No data available. Please load data first.")
                
            # Initialize and fit K-Means
            kmeans = KMeans(n_clusters=n_clusters, 
                        random_state=random_state,
                        n_init=10)  # Explicitly set n_init to avoid warning
            cluster_labels = kmeans.fit_predict(self.normalized_data)
            
            # Calculate evaluation metrics
            silhouette = silhouette_score(self.normalized_data, cluster_labels)
            inertia = kmeans.inertia_
            
            # Store results
            self.results['kmeans'] = {
                'model': kmeans,
                'labels': cluster_labels,
                'metrics': {
                    'silhouette_score': silhouette,
                    'inertia': inertia,
                    'n_clusters': n_clusters
                }
            }
            
            # Visualization
            if visualize and len(self.features) > 1:
                self._visualize_clusters(cluster_labels, algorithm='K-Means')
            
            print(f"K-Means clustering completed with {n_clusters} clusters")
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Inertia: {inertia:.3f}")
            
            return self.results['kmeans']
            
        except Exception as e:
            print(f"Error in K-Means clustering: {str(e)}")
            return None
        
    

    
    def run_kmedoids(self, n_clusters=3, max_iter=300, random_state=None, visualize=True):
        """
        K-Medoids clustering using PAM algorithm
        
        Parameters:
        -----------
        n_clusters : int (default=3)
            Number of clusters to form
        max_iter : int (default=300)
            Maximum number of iterations
        random_state : int (optional)
            Random seed for reproducibility
        visualize : bool (default=True)
            Whether to plot the clusters
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'model': Dictionary with medoid indices
            - 'labels': Cluster assignments
            - 'metrics': Evaluation metrics
        """
        try:
            if self.normalized_data is None:
                raise ValueError("No data available. Please load data first.")
                
            X = self.normalized_data
            n_samples = X.shape[0]
            
            # Initialize random seed if provided
            if random_state is not None:
                np.random.seed(random_state)
            
            # 1. Randomly select initial medoids
            medoid_indices = np.random.choice(n_samples, n_clusters, replace=False)
            medoids = X[medoid_indices]
            
            # 2. Assign points to closest medoid
            distances = pairwise_distances(X, medoids)
            labels = np.argmin(distances, axis=1)
            
            # PAM algorithm
            for _ in range(max_iter):
                # 3. For each medoid m and non-medoid o, calculate cost of swapping
                best_medoids = medoid_indices.copy()
                improved = False
                
                for i in range(n_clusters):
                    # Current medoid
                    m = medoid_indices[i]
                    
                    # Get all non-medoid points in this cluster
                    cluster_points = np.where(labels == i)[0]
                    non_medoids = np.setdiff1d(cluster_points, medoid_indices)
                    
                    if len(non_medoids) == 0:
                        continue
                    
                    # Calculate current cost (total distance to medoid)
                    current_cost = np.sum(distances[cluster_points, i])
                    
                    # Try swapping with each non-medoid
                    for o in non_medoids:
                        # Temporary swap
                        temp_medoids = medoid_indices.copy()
                        temp_medoids[i] = o
                        
                        # Calculate new distances and labels
                        temp_distances = pairwise_distances(X, X[temp_medoids])
                        temp_labels = np.argmin(temp_distances, axis=1)
                        
                        # Calculate new cost
                        new_cost = np.sum([temp_distances[j, temp_labels[j]] 
                                         for j in range(n_samples)])
                        
                        # If better, keep this swap
                        if new_cost < current_cost:
                            best_medoids = temp_medoids.copy()
                            current_cost = new_cost
                            improved = True
                
                # Check for convergence
                if not improved:
                    break
                    
                # Update medoids and distances
                medoid_indices = best_medoids
                distances = pairwise_distances(X, X[medoid_indices])
                labels = np.argmin(distances, axis=1)
            
            # Calculate final metrics
            silhouette = silhouette_score(X, labels)
            inertia = np.sum([distances[i, labels[i]] for i in range(n_samples)])
            
            # Store results
            self.results['kmedoids'] = {
                'model': {
                    'medoid_indices_': medoid_indices,
                    'medoids_': X[medoid_indices]
                },
                'labels': labels,
                'metrics': {
                    'silhouette_score': silhouette,
                    'inertia': inertia,
                    'n_clusters': n_clusters
                }
            }
            
            if visualize and len(self.features) > 1:
                self._visualize_clusters(labels, algorithm='K-Medoids')
            
            print(f"K-Medoids clustering completed with {n_clusters} clusters")
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Inertia: {inertia:.3f}")
            print(f"Medoid indices: {medoid_indices}")
            
            return self.results['kmedoids']
            
        except Exception as e:
            print(f"Error in K-Medoids clustering: {str(e)}")
            return None
    
    def run_agnes(self, n_clusters=3, linkage='ward', visualize=True, plot_dendrogram=False):
        """
        Parameters:
        -----------
        n_clusters : int (default=3)
            Number of clusters to form
        linkage : str (default='ward')
            Linkage criterion ('ward', 'complete', 'average', 'single')
        visualize : bool (default=True)
            Whether to plot the clusters
        plot_dendrogram : bool (default=False)
            Whether to plot the full dendrogram
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'model': The trained AgglomerativeClustering model
            - 'labels': Cluster assignments
            - 'metrics': Evaluation metrics
        """
        try:
            if self.normalized_data is None:
                raise ValueError("No data available. Please load data first.")
                
            agnes = AgglomerativeClustering(n_clusters=n_clusters,
                                          linkage=linkage)
            cluster_labels = agnes.fit_predict(self.normalized_data)
            
            silhouette = silhouette_score(self.normalized_data, cluster_labels)
            
            self.results['agnes'] = {
                'model': agnes,
                'labels': cluster_labels,
                'metrics': {
                    'silhouette_score': silhouette,
                    'n_clusters': n_clusters,
                    'linkage': linkage
                }
            }
            
            if visualize and len(self.features) > 1:
                self._visualize_clusters(cluster_labels, algorithm='AGNES')
                
            if plot_dendrogram:
                self._plot_dendrogram()
            
            print(f"AGNES clustering completed with {n_clusters} clusters")
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Linkage method: {linkage}")
            
            return self.results['agnes']
            
        except Exception as e:
            print(f"Error in AGNES clustering: {str(e)}")
            return None
    
    def _plot_dendrogram(self, **kwargs):
        """
        Plot the hierarchical clustering dendrogram
        """
        plt.figure(figsize=(10, 7))
        plt.title("Hierarchical Clustering Dendrogram")
        
        # Calculate linkage matrix
        Z = linkage(self.normalized_data, 'ward')
        
        # Plot dendrogram
        dendrogram(Z, **kwargs)
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        plt.show()
    
    # def run_diana(self, n_clusters):
       
    #     pass
    
    # def run_dbscan(self, eps=0.5, min_samples=5):
       
    #     pass
    
    # def evaluate_clusters(self, algorithm):
        
    #     pass
    
    # def plot_elbow_method(self):
       
    #     pass
    
    # def visualize_clusters(self):
       
    #     pass

    def _visualize_clusters(self, labels, algorithm=''):
        """Internal method to visualize clusters using PCA (2D projection)"""
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.normalized_data)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced_data[:, 0], 
                            reduced_data[:, 1], 
                            c=labels, 
                            cmap='viridis',
                            alpha=0.7)
        
        plt.title(f'{algorithm} Clustering (PCA-reduced)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()
    


app = ClusteringApp()
# example usage 
if __name__ == "__main__":
    app = ClusteringApp()
    filepath = "insur_3_new.csv"
    
    if app.load_data(filepath, numeric_only=False):
        print("Data loaded successfully.")
        print("Data preview:")   
        
        # Run K-Means
        kmeans_result = app.run_kmeans(n_clusters=3)
        if kmeans_result:
            print("\nK-Means results:")
            print("Cluster assignments:", kmeans_result['labels'])
            print("Model parameters:", kmeans_result['model'].get_params())
        
        # Run K-Medoids
        kmedoids_result = app.run_kmedoids(n_clusters=3)
        if kmedoids_result:
            print("\nK-Medoids results:")
            print("Cluster assignments:", kmedoids_result['labels'])
            print("Medoid indices:", kmedoids_result['model'].medoid_indices_)
        
        # Run AGNES
        agnes_result = app.run_agnes(n_clusters=3, plot_dendrogram=True)
        if agnes_result:
            print("\nAGNES results:")
            print("Cluster assignments:", agnes_result['labels'])
            print("Linkage method:", agnes_result['metrics']['linkage'])
    else:
        print("Failed to load data")
