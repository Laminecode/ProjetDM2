from imports import *

class ClusteringApp:
    def __init__(self):
        self.raw_data = None
        self.data = None
        self.normalized_data = None
        self.features = None
        self.scaler = None
        self.label_encoders = {}
        self.results = {}
        
    def load_data(self, filepath, numeric_only=False, sample_size=None, selected_features=None):
        """
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        numeric_only : bool
            If True, keeps only numeric columns
        sample_size : int (optional)
            Number of instances to randomly sample
        selected_features : list (optional)
            List of features to keep
            
        Returns:
        --------
        bool
            True if successful, False if error occurred
        """
        try:
            # Read CSV, converting '?' to NaN
            self.raw_data = pd.read_csv(filepath, na_values='?')
            self.label_encoders = {}
            
            # Apply sampling if requested
            if sample_size is not None and sample_size < len(self.raw_data):
                self.raw_data = self.raw_data.sample(n=sample_size, random_state=42)
            
            # Select features if requested
            if selected_features is not None:
                available_features = [f for f in selected_features if f in self.raw_data.columns]
                self.raw_data = self.raw_data[available_features]
            
            if numeric_only:
                # Keep only numeric columns
                self.data = self.raw_data.select_dtypes(include=['number'])
            else:
                # Create a copy to avoid modifying raw data
                self.data = self.raw_data.copy()
                
                # Process non-numeric columns
                for col in self.raw_data.select_dtypes(exclude=['number']).columns:
                    # Convert to string and handle missing values
                    self.data[col] = self.data[col].astype(str).str.lower().str.strip()
                    # Replace '?' strings that weren't caught by na_values
                    self.data[col] = self.data[col].replace('?', np.nan)
                    
                    # Fill missing categorical values with 'missing' before encoding
                    self.data[col] = self.data[col].fillna('missing')
                    
                    # Label encoding
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col])
                    self.label_encoders[col] = le
                    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            
            # Drop rows with any remaining missing values in numeric columns
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


    def find_optimal_clusters(self, method='elbow', max_clusters=7, algorithm='kmeans'):
        """
        Parameters:
        -----------
        method : str ('elbow' or 'silhouette')
            Method to use for determining optimal clusters
        max_clusters : int
            Maximum number of clusters to try
        algorithm : str
            Clustering algorithm to use ('kmeans', 'kmedoids', 'agnes', 'diana')
            
        Returns:
        --------
        int
            Suggested number of clusters
        """
        if self.normalized_data is None:
            raise ValueError("No data available. Please load data first.")
            
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters+1)
        
        for k in k_range:
            if algorithm == 'kmeans':
                result = self.run_kmeans(n_clusters=k, visualize=False)
                inertias.append(result['metrics']['inertia'])
            elif algorithm == 'kmedoids':
                result = self.run_kmedoids(n_clusters=k, visualize=False)
                inertias.append(result['metrics']['inertia'])
            elif algorithm == 'agnes':
                result = self.run_agnes(n_clusters=k, visualize=False)
                # AGNES doesn't have inertia, so we'll use pairwise distances
                distances = pdist(self.normalized_data)
                Z = linkage(distances, 'ward')
                labels = fcluster(Z, k, criterion='maxclust')
                cluster_distances = []
                for i in range(1, k+1):
                    cluster_points = self.normalized_data[labels == i]
                    if len(cluster_points) > 0:
                        centroid = np.mean(cluster_points, axis=0)
                        cluster_distances.append(np.sum(np.linalg.norm(cluster_points - centroid, axis=1)**2))
                inertias.append(np.sum(cluster_distances))
            elif algorithm == 'diana':
                result = self.run_diana(n_clusters=k, visualize=False)
                # Calculate inertia-like metric for DIANA
                labels = result['labels']
                cluster_distances = []
                for i in range(k):
                    cluster_points = self.normalized_data[labels == i]
                    if len(cluster_points) > 0:
                        centroid = np.mean(cluster_points, axis=0)
                        cluster_distances.append(np.sum(np.linalg.norm(cluster_points - centroid, axis=1)**2))
                inertias.append(np.sum(cluster_distances))
            
            silhouette_scores.append(result['metrics']['silhouette_score'])
        
        # Plot results
        plt.figure(figsize=(12, 5))
        
        if method == 'elbow':
            plt.subplot(1, 2, 1)
            plt.plot(k_range, inertias, 'bo-')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            plt.title(f'Elbow Method ({algorithm.upper()})')
            
            # Find the elbow point
            diffs = np.diff(inertias)
            diff_ratios = diffs[:-1] / diffs[1:]
            optimal_k = np.argmax(diff_ratios) + 2  # +2 because we start from k=2
            
        elif method == 'silhouette':
            plt.subplot(1, 2, 1)
            plt.plot(k_range, silhouette_scores, 'bo-')
            plt.xlabel('Number of clusters')
            plt.ylabel('Silhouette Score')
            plt.title(f'Silhouette Method ({algorithm.upper()})')
            optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Plot both metrics for reference
        plt.subplot(1, 2, 2)
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title(f'Inertia across cluster numbers ({algorithm.upper()})')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Suggested number of clusters for {algorithm.upper()}: {optimal_k}")
        return optimal_k

    def _calculate_intra_class_distance(self, data, labels):
        intra_distances = []
        for cluster in np.unique(labels):
            cluster_points = data[labels == cluster]
            if len(cluster_points) > 1:
                distances = pdist(cluster_points)
                intra_distances.append(np.mean(distances))
        return np.mean(intra_distances) if intra_distances else 0


    def _calculate_inter_class_distance(self, data, labels):
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0  
        
        centroids = []
        for cluster in unique_labels:
            cluster_points = data[labels == cluster]
            centroids.append(np.mean(cluster_points, axis=0))
        
        inter_distances = pdist(centroids)
        return np.mean(inter_distances) if len(inter_distances) > 0 else 0
    

    def run_kmeans(self, n_clusters=3, random_state=42, visualize=True):
        """
        Run K-Means clustering algorithm
        
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
            if self.normalized_data is None:
                raise ValueError("No data available. Please load data first.")
                
            start_time = timer()
            kmeans = KMeans(n_clusters=n_clusters, 
                          random_state=random_state,
                          n_init=10)
            cluster_labels = kmeans.fit_predict(self.normalized_data)
            
            silhouette = silhouette_score(self.normalized_data, cluster_labels)
            inertia = kmeans.inertia_
            ch_score = calinski_harabasz_score(self.normalized_data, cluster_labels)
            db_score = davies_bouldin_score(self.normalized_data, cluster_labels)
            intra_class = self._calculate_intra_class_distance(self.normalized_data, cluster_labels)
            inter_class = self._calculate_inter_class_distance(self.normalized_data, cluster_labels)

            end_time = timer()
            execution_time = end_time - start_time
            self.results['kmeans'] = {
                'model': kmeans,
                'labels': cluster_labels,
                'metrics': {
                    'silhouette_score': silhouette,
                    'inertia': inertia,
                    'calinski_harabasz': ch_score,
                    'davies_bouldin': db_score,
                    'intra_class_distance': intra_class,
                    'inter_class_distance': inter_class,
                    'n_clusters': n_clusters,
                    'execution_time': execution_time
                }
            }     
            
            if visualize and len(self.features) > 1:
                self._visualize_clusters(cluster_labels, algorithm='K-Means')
            
            print(f"K-Means clustering completed with {n_clusters} clusters")
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Inertia: {inertia:.3f}")
            print(f"Calinski-Harabasz Index: {ch_score:.3f}")
            print(f"Davies-Bouldin Index: {db_score:.3f}")
            
            return self.results['kmeans']
            
        except Exception as e:
            print(f"Error in K-Means clustering: {str(e)}")
            return None
        
    def run_kmedoids(self, n_clusters=3, max_iter=300, random_state=None, visualize=True):
        """
        
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

            start_time = timer() 
            X = self.normalized_data
            n_samples = X.shape[0]
            
            if random_state is not None:
                np.random.seed(random_state)
            
            # Initialize medoids randomly
            medoid_indices = np.random.choice(n_samples, n_clusters, replace=False)
            distances = pairwise_distances(X, X[medoid_indices])
            labels = np.argmin(distances, axis=1)
            
            for _ in range(max_iter):
                # Calculate current cost
                current_cost = np.sum(np.min(distances, axis=1))
                
                best_medoids = medoid_indices.copy()
                improved = False
                
                # For each cluster
                for i in range(n_clusters):
                    # Get all non-medoid points in this cluster
                    cluster_points = np.where(labels == i)[0]
                    non_medoids = np.setdiff1d(cluster_points, medoid_indices)
                    
                    if len(non_medoids) == 0:
                        continue
                    
                    # Try swapping with each non-medoid
                    for o in non_medoids:
                        # Create temporary medoids with this swap
                        temp_medoids = medoid_indices.copy()
                        temp_medoids[i] = o
                        
                        # Calculate new distances
                        temp_distances = pairwise_distances(X, X[temp_medoids])
                        temp_labels = np.argmin(temp_distances, axis=1)
                        new_cost = np.sum(np.min(temp_distances, axis=1))
                        
                        # If better, keep this swap
                        if new_cost < current_cost:
                            best_medoids = temp_medoids
                            current_cost = new_cost
                            improved = True
                
                if not improved:
                    break
                    
                # Update medoids and distances
                medoid_indices = best_medoids
                distances = pairwise_distances(X, X[medoid_indices])
                labels = np.argmin(distances, axis=1)
            
            # Calculate final metrics
            silhouette = silhouette_score(X, labels)
            inertia = np.sum(np.min(distances, axis=1))
            ch_score = calinski_harabasz_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            intra_class = self._calculate_intra_class_distance(self.normalized_data, labels)
            inter_class = self._calculate_inter_class_distance(self.normalized_data, labels)

            end_time = timer()
            execution_time = end_time - start_time
            self.results['kmedoids'] = {
                'model': {
                    'medoid_indices_': medoid_indices,
                    'medoids_': X[medoid_indices]
                },
                'labels': labels,
                'metrics': {
                    'silhouette_score': silhouette,
                    'inertia': inertia,
                    'calinski_harabasz': ch_score,
                    'davies_bouldin': db_score,
                    'intra_class_distance': intra_class,
                    'inter_class_distance': inter_class,
                    'n_clusters': n_clusters,
                    'execution_time': execution_time
                }
            }
            
            if visualize and len(self.features) > 1:
                self._visualize_clusters(labels, algorithm='K-Medoids')
            
            print(f"K-Medoids clustering completed with {n_clusters} clusters")
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Inertia: {inertia:.3f}")
            print(f"Calinski-Harabasz Index: {ch_score:.3f}")
            print(f"Davies-Bouldin Index: {db_score:.3f}")
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
            
            start_time = timer()
            agnes = AgglomerativeClustering(n_clusters=n_clusters,
                                         linkage=linkage)
            cluster_labels = agnes.fit_predict(self.normalized_data)
            
            silhouette = silhouette_score(self.normalized_data, cluster_labels)
            ch_score = calinski_harabasz_score(self.normalized_data, cluster_labels)
            db_score = davies_bouldin_score(self.normalized_data, cluster_labels)
            intra_class = self._calculate_intra_class_distance(self.normalized_data, cluster_labels)
            inter_class = self._calculate_inter_class_distance(self.normalized_data, cluster_labels)
            
            end_time = timer()
            execution_time = end_time - start_time
            self.results['agnes'] = {
                'model': agnes,
                'labels': cluster_labels,
                'metrics': {
                    'silhouette_score': silhouette,
                    'calinski_harabasz': ch_score,
                    'davies_bouldin': db_score,
                    'intra_class_distance': intra_class,
                    'inter_class_distance': inter_class,
                    'n_clusters': n_clusters,
                    'linkage': linkage,
                    'execution_time': execution_time
                }
            }
            if visualize and len(self.features) > 1:
                self._visualize_clusters(cluster_labels, algorithm='AGNES')
                
            if plot_dendrogram:
                self._plot_dendrogram()
            
            print(f"AGNES clustering completed with {n_clusters} clusters")
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Calinski-Harabasz Index: {ch_score:.3f}")
            print(f"Davies-Bouldin Index: {db_score:.3f}")
            print(f"Linkage method: {linkage}")
            
            return self.results['agnes']
            
        except Exception as e:
            print(f"Error in AGNES clustering: {str(e)}")
            return None
    
    def run_diana(self, n_clusters=3, visualize=True, plot_dendrogram=False):
        """
        
        Parameters:
        -----------
        n_clusters : int (default=3)
            Number of clusters to form
        visualize : bool (default=True)
            Whether to plot the clusters
        plot_dendrogram : bool (default=False)
            Whether to plot the full dendrogram
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'model': Dictionary with clustering information
            - 'labels': Cluster assignments
            - 'metrics': Evaluation metrics
        """
        try:
            if self.normalized_data is None:
                raise ValueError("No data available. Please load data first.")

            start_time = timer()   
            X = self.normalized_data
            n_samples = X.shape[0]
            
            # Start with all points in one cluster
            clusters = [list(range(n_samples))]
            
            while len(clusters) < n_clusters:
                # Find the cluster with largest diameter to split
                max_diameter = -1
                cluster_to_split = None
                
                for i, cluster in enumerate(clusters):
                    if len(cluster) < 2:
                        continue
                    
                    # Calculate diameter (maximum distance between any two points)
                    distances = pairwise_distances(X[cluster])
                    diameter = np.max(distances)
                    
                    if diameter > max_diameter:
                        max_diameter = diameter
                        cluster_to_split = i
                
                if cluster_to_split is None:
                    break  # Can't split any further
                
                # Split the selected cluster
                cluster = clusters.pop(cluster_to_split)
                
                # Find the most dissimilar pair as seeds for new clusters
                distances = pairwise_distances(X[cluster])
                max_dist_indices = np.unravel_index(np.argmax(distances), distances.shape)
                seed1, seed2 = max_dist_indices
                
                new_cluster1 = [cluster[seed1]]
                new_cluster2 = [cluster[seed2]]
                
                # Assign remaining points to the closest seed
                remaining_points = [p for i, p in enumerate(cluster) if i not in {seed1, seed2}]
                
                for point in remaining_points:
                    dist1 = np.linalg.norm(X[point] - X[cluster[seed1]])
                    dist2 = np.linalg.norm(X[point] - X[cluster[seed2]])
                    
                    if dist1 < dist2:
                        new_cluster1.append(point)
                    else:
                        new_cluster2.append(point)
                
                # Add the new clusters
                clusters.append(new_cluster1)
                clusters.append(new_cluster2)
            
            # Create labels array
            labels = np.zeros(n_samples, dtype=int)
            for i, cluster in enumerate(clusters):
                for point in cluster:
                    labels[point] = i
            
            # Calculate metrics
            silhouette = silhouette_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            intra_class = self._calculate_intra_class_distance(X, labels)
            inter_class = self._calculate_inter_class_distance(X, labels)

            end_time = timer()
            execution_time = end_time - start_time
            self.results['diana'] = {
                'model': {
                    'clusters_': clusters,
                    'n_clusters_': len(clusters)
                },
                'labels': labels,
                'metrics': {
                    'silhouette_score': silhouette,
                    'calinski_harabasz': ch_score,
                    'davies_bouldin': db_score,
                    'intra_class_distance': intra_class,
                    'inter_class_distance': inter_class,
                    'n_clusters': len(clusters),
                    'execution_time': execution_time
                }
            }
            if visualize and len(self.features) > 1:
                self._visualize_clusters(labels, algorithm='DIANA')
                
            if plot_dendrogram:
                self._plot_dendrogram()
            
            print(f"DIANA clustering completed with {len(clusters)} clusters")
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Calinski-Harabasz Index: {ch_score:.3f}")
            print(f"Davies-Bouldin Index: {db_score:.3f}")
            
            return self.results['diana']
            
        except Exception as e:
            print(f"Error in DIANA clustering: {str(e)}")
            return None
    
    def run_dbscan(self, eps=0.5, min_samples=5, visualize=True):
        """
        
        Parameters:
        -----------
        eps : float (default=0.5)
            Maximum distance between two samples for one to be considered in the neighborhood of the other
        min_samples : int (default=5)
            Number of samples in a neighborhood for a point to be considered a core point
        visualize : bool (default=True)
            Whether to plot the clusters
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'model': The trained DBSCAN model
            - 'labels': Cluster assignments
            - 'metrics': Evaluation metrics
        """
        try:
            if self.normalized_data is None:
                raise ValueError("No data available. Please load data first.")
                
            start_time = timer()
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(self.normalized_data)
            
            # Calculate metrics (only if there are clusters found)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            if n_clusters > 0:
                # Filter out noise points for metrics calculation
                filtered_data = self.normalized_data[cluster_labels != -1]
                filtered_labels = cluster_labels[cluster_labels != -1]
                intra_class = self._calculate_intra_class_distance(filtered_data, filtered_labels)
                inter_class = self._calculate_inter_class_distance(filtered_data, filtered_labels)
                if len(set(filtered_labels)) > 1:  # Need at least 2 clusters for these metrics
                    silhouette = silhouette_score(filtered_data, filtered_labels)
                    ch_score = calinski_harabasz_score(filtered_data, filtered_labels)
                    db_score = davies_bouldin_score(filtered_data, filtered_labels)
                else:
                    silhouette = -1  # Invalid value
                    ch_score = -1
                    db_score = -1
            else:
                silhouette = -1
                ch_score = -1
                db_score = -1
                intra_class = -1
                inter_class = -1
            
            end_time = timer()
            execution_time = end_time - start_time
            self.results['dbscan'] = {
                'model': dbscan,
                'labels': cluster_labels,
                'metrics': {
                    'silhouette_score': silhouette,
                    'calinski_harabasz': ch_score,
                    'davies_bouldin': db_score,
                    'intra_class_distance': intra_class,
                    'inter_class_distance': inter_class,
                    'n_clusters': n_clusters,
                    'noise_points': np.sum(cluster_labels == -1),
                    'eps': eps,
                    'min_samples': min_samples,
                    'execution_time': execution_time
                }
            }   
            
            if visualize and len(self.features) > 1:
                self._visualize_clusters(cluster_labels, algorithm='DBSCAN')
            
            print(f"DBSCAN clustering completed with {n_clusters} clusters")
            print(f"Silhouette Score: {silhouette:.3f}" if silhouette != -1 else "Silhouette Score: Not computable")
            print(f"Calinski-Harabasz Index: {ch_score:.3f}" if ch_score != -1 else "Calinski-Harabasz Index: Not computable")
            print(f"Davies-Bouldin Index: {db_score:.3f}" if db_score != -1 else "Davies-Bouldin Index: Not computable")
            print(f"Noise points: {np.sum(cluster_labels == -1)}")
            
            return self.results['dbscan']
            
        except Exception as e:
            print(f"Error in DBSCAN clustering: {str(e)}")
            return None
    
    def _plot_dendrogram(self, **kwargs):
        """
        Plot hierarchical clustering dendrogram
        """
        plt.figure(figsize=(10, 7))
        plt.title("Hierarchical Clustering Dendrogram")
        
        Z = linkage(self.normalized_data, 'ward')
        dendrogram(Z, **kwargs)
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        plt.show()
    
    def _visualize_clusters(self, labels, algorithm=''):
        """
        Visualize clusters using PCA for dimensionality reduction
        """
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
    
    def compare_algorithms(self, n_clusters=3, n_clusters_diana=None, n_clusters_kmeans=None,n_clusters_kmedoids=None, n_clusters_agnes=None):
        """
        Parameters:
        -----------
        n_clusters : int (default=3)
            Default number of clusters for all algorithms
        n_clusters_diana : int (optional)
            Number of clusters for DIANA (overrides default)
        n_clusters_kmeans : int (optional)
            Number of clusters for K-Means (overrides default)
        n_clusters_kmedoids : int (optional)
            Number of clusters for K-Medoids (overrides default)
        n_clusters_agnes : int (optional)
            Number of clusters for AGNES (overrides default)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with comparison metrics
        """
        # Input validation
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        
        # Set default values if not provided
        n_clusters_diana = n_clusters_diana if n_clusters_diana is not None else n_clusters
        n_clusters_kmeans = n_clusters_kmeans if n_clusters_kmeans is not None else n_clusters
        n_clusters_kmedoids = n_clusters_kmedoids if n_clusters_kmedoids is not None else n_clusters
        n_clusters_agnes = n_clusters_agnes if n_clusters_agnes is not None else n_clusters

        # Validate all cluster counts
        for k, name in [(n_clusters_diana, 'DIANA'),
                    (n_clusters_kmeans, 'K-Means'),
                    (n_clusters_kmedoids, 'K-Medoids'),
                    (n_clusters_agnes, 'AGNES')]:
            if not isinstance(k, int) or k < 1:
                raise ValueError(f"{name} cluster count must be a positive integer")

        # Clear previous results
        self.results = {}
        
        # Run all algorithms
        self.run_kmeans(n_clusters=n_clusters_kmeans, visualize=False)
        self.run_kmedoids(n_clusters=n_clusters_kmedoids, visualize=False)
        self.run_agnes(n_clusters=n_clusters_agnes, visualize=False)
        self.run_diana(n_clusters=n_clusters_diana, visualize=False)
        dbscan_result = self.run_dbscan(visualize=False)
        
        # Collect metrics
        metrics = []
        for algo, result in self.results.items():
            if algo == 'dbscan':
                # Special handling for DBSCAN
                if result['metrics']['n_clusters'] == 0:
                    continue
                # Filter out noise points for metrics calculation
                mask = result['labels'] != -1
                filtered_data = self.normalized_data[mask]
                filtered_labels = result['labels'][mask]
                if len(np.unique(filtered_labels)) < 2:
                    continue  # Need at least 2 clusters for meaningful metrics
            else:
                filtered_data = self.normalized_data
                filtered_labels = result['labels']
            
            # Calculate metrics
            silhouette = silhouette_score(filtered_data, filtered_labels)
            ch_score = calinski_harabasz_score(filtered_data, filtered_labels)
            db_score = davies_bouldin_score(filtered_data, filtered_labels)
            intra_class = self._calculate_intra_class_distance(filtered_data, filtered_labels)
            inter_class = self._calculate_inter_class_distance(filtered_data, filtered_labels)
            
            metrics.append({
                'Algorithm': algo.upper(),
                'Silhouette': silhouette,
                'Calinski-Harabasz': ch_score,
                'Davies-Bouldin': db_score,
                'Inertia': result['metrics'].get('inertia', np.nan),
                'Intra-Class': intra_class,
                'Inter-Class': inter_class,
                'Ratio (I/C)': inter_class / intra_class if intra_class > 0 else np.inf,
                'Time (s)': result['metrics']['execution_time'],
                'Clusters': result['metrics']['n_clusters'],
                'Noise Points': result['metrics'].get('noise_points', 0)
            })
        
        # Rest of the method remains the same...
        # Create and display comparison table
        df = pd.DataFrame(metrics)
        if df.empty:
            print("Warning: No valid clustering results to compare")
            return df
        
        df.set_index('Algorithm', inplace=True)
        
        # Plot comparison
        plt.figure(figsize=(18, 6))
        
        # Intra/Inter Class comparison
        plt.subplot(1, 4, 1)
        df[['Intra-Class', 'Inter-Class']].plot(kind='bar', ax=plt.gca())
        plt.title('Intra vs Inter Class Distance')
        plt.ylabel('Distance')
        
        # Ratio comparison
        plt.subplot(1, 4, 2)
        df['Ratio (I/C)'].plot(kind='bar', color='purple')
        plt.title('Inter/Intra Class Ratio')
        plt.ylabel('Ratio (Higher is better)')
        
        # Silhouette Score comparison
        plt.subplot(1, 4, 3)
        df['Silhouette'].plot(kind='bar', color='skyblue')
        plt.title('Silhouette Score Comparison')
        plt.ylim(0, 1)
        
        # Execution time comparison
        plt.subplot(1, 4, 4)
        df['Time (s)'].plot(kind='bar', color='orange')
        plt.title('Execution Time Comparison')
        plt.ylabel('Seconds')
        
        plt.tight_layout()
        plt.show()
        
        # Print timing information
        print("\nExecution Time Summary:")
        time_df = df[['Time (s)']].copy()
        time_df['Relative Time'] = time_df['Time (s)'] / time_df['Time (s)'].min()
        print(time_df.sort_values('Time (s)'))
        
        return df
if __name__ == "__main__":
    app = ClusteringApp()
    filepath = "insur_3_new.csv"
    
    if app.load_data(filepath, numeric_only=False, sample_size=500):
        print("Data loaded successfully.")
        print("Data preview:")   
        # print(app.data.head())
        
        # Find optimal clusters for K-Means
        # optimal_k1 = app.find_optimal_clusters(method='elbow', algorithm='kmeans')
        # optimal_k2 = app.find_optimal_clusters(method='elbow', algorithm='kmedoids')
        # optimal_k3 = app.find_optimal_clusters(method='elbow', algorithm='agnes')
        # optimal_k4 = app.find_optimal_clusters(method='elbow', algorithm='diana')
        
        # app.run_agnes(n_clusters=3, visualize=True, plot_dendrogram=True)
        # comparison = app.compare_algorithms(n_clusters_kmeans=optimal_k1, n_clusters_kmedoids=optimal_k2, n_clusters_agnes=optimal_k3, n_clusters_diana=optimal_k4)
        # comparison = app.compare_algorithms()
        # print("\nAlgorithm Comparison:")
        # print(comparison)
        app.run_agnes(n_clusters=3, visualize=True, plot_dendrogram=True)
    else:
        print("Failed to load data")