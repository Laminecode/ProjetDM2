import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QComboBox, QTextEdit, QProgressBar, QFrame, QStackedWidget,
                            QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
                            QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QRadioButton,
                            QButtonGroup, QMessageBox, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
<<<<<<< HEAD
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
=======
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances
>>>>>>> c261d8f72717591b2d1a421c5fa94976f9e6dc28

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from timeit import default_timer as timer

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)


class UploadAndPreprocess(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.data = None
        self.raw_data = None
        self.label_encoders = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create navbar with tabs
        self.tabs = QTabWidget()

        # Add tabs
        self.upload_tab = QWidget()
        self.clean_data_tab = QWidget()
        self.encoding_tab = QWidget()
        self.normalization_tab = QWidget()

        self.tabs.addTab(self.upload_tab, "Upload")
        self.tabs.addTab(self.clean_data_tab, "Clean Data")
        self.tabs.addTab(self.encoding_tab, "Encoding")
        self.tabs.addTab(self.normalization_tab, "Normalization")

        # Initialize tabs
        self.init_upload_tab()
        self.init_clean_data_tab()
        self.init_encoding_tab()
        self.init_normalization_tab()

        layout.addWidget(self.tabs)

    def init_upload_tab(self):
        layout = QVBoxLayout(self.upload_tab)

        upload_options = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        upload_button = QPushButton("Upload Data")
        upload_button.clicked.connect(self.upload_file)
        
        self.sample_size_label = QLabel("Sample size:")
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setMinimum(0)
        self.sample_size_spin.setMaximum(1000000)
        self.sample_size_spin.setValue(0)
        self.sample_size_spin.setSpecialValueText("All")
        
        self.numeric_only_cb = QCheckBox("Numeric only")

        upload_options.addWidget(upload_button)
        upload_options.addWidget(self.file_label)
        upload_options.addWidget(self.sample_size_label)
        upload_options.addWidget(self.sample_size_spin)
        upload_options.addWidget(self.numeric_only_cb)
        
        layout.addLayout(upload_options)

        # Table to display data
        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)

    def init_clean_data_tab(self):
        layout = QVBoxLayout(self.clean_data_tab)

        options_layout = QHBoxLayout()
        clean_missing_button = QPushButton("Remove Missing Values")
        clean_missing_button.clicked.connect(self.clean_data)
        
        options_layout.addWidget(clean_missing_button)
        layout.addLayout(options_layout)

        # Table to display cleaned data
        self.clean_data_table = QTableWidget()
        layout.addWidget(self.clean_data_table)

    def init_encoding_tab(self):
        layout = QVBoxLayout(self.encoding_tab)

        encode_button = QPushButton("Encode Categorical Variables")
        encode_button.clicked.connect(self.encode_categorical)

        layout.addWidget(encode_button)

        # Table to display encoded data
        self.encoding_table = QTableWidget()
        layout.addWidget(self.encoding_table)

    def init_normalization_tab(self):
        layout = QVBoxLayout(self.normalization_tab)

        normalize_button = QPushButton("Apply Normalization")
        normalize_button.clicked.connect(self.apply_normalization)

        layout.addWidget(normalize_button)

        # Table to display normalized data
        self.normalization_table = QTableWidget()
        layout.addWidget(self.normalization_table)

    def upload_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "Data Files (*.csv);;All Files (*)"
        )

        if filename:
            self.file_label.setText(f"Selected: {filename}")
            
            # Get sample size and numeric only setting
            sample_size = self.sample_size_spin.value() if self.sample_size_spin.value() > 0 else None
            numeric_only = self.numeric_only_cb.isChecked()
            
            try:
                # Load data similar to the ClusteringApp.load_data method
                self.raw_data = pd.read_csv(filename, na_values='?')
                self.label_encoders = {}
                
                # Apply sampling if requested
                if sample_size is not None and sample_size < len(self.raw_data):
                    self.raw_data = self.raw_data.sample(n=sample_size, random_state=42)
                
                if numeric_only:
                    # Keep only numeric columns
                    self.data = self.raw_data.select_dtypes(include=['number'])
                else:
                    # Create a copy to avoid modifying raw data
                    self.data = self.raw_data.copy()
                
                self.update_table(self.data_table, self.data)
                
                if self.parent:
                    self.parent.data = self.data
                    self.parent.raw_data = self.raw_data
                    self.parent.update_feature_selectors()
                
                QMessageBox.information(self, "Success", "Data loaded successfully.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
    
    def clean_data(self):
        if self.data is not None:
            try:
                # Fill missing values for each column
                for column in self.data.columns:
                    if self.data[column].dtype in ['float64', 'int64']:  # Numerical columns
                        self.data[column].fillna(self.data[column].mean(), inplace=True)
                    else:  # Categorical columns
                        self.data[column].fillna(self.data[column].mode()[0], inplace=True)
                
                # Drop any remaining rows with missing values
                self.data = self.data.dropna()
                
                # Update the table to display cleaned data
                self.update_table(self.clean_data_table, self.data)
                
                if self.parent:
                    self.parent.data = self.data
                    
                QMessageBox.information(self, "Success", "Data cleaned successfully.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error cleaning data: {str(e)}")
    
    def encode_categorical(self):
        if self.data is not None:
            try:
                # Process non-numeric columns with label encoding
                for col in self.data.select_dtypes(exclude=['number']).columns:
                    # Convert to string and handle missing values
                    self.data[col] = self.data[col].astype(str).str.lower().str.strip()
                    # Replace '?' strings that weren't caught by na_values
                    self.data[col] = self.data[col].replace('?', np.nan)
                    
                    # Fill missing categorical values with 'missing'
                    self.data[col] = self.data[col].fillna('missing')
                    
                    # Label encoding
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col])
                    self.label_encoders[col] = le
                
                # Update the table
                self.update_table(self.encoding_table, self.data)
                
                if self.parent:
                    self.parent.data = self.data
                    self.parent.label_encoders = self.label_encoders
                
                QMessageBox.information(self, "Success", "Categorical variables encoded successfully.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error encoding categorical variables: {str(e)}")
    
    def apply_normalization(self):
        if self.data is not None:
            try:
                # Normalize numeric columns
                numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
                
                if not numeric_data.empty:
                    scaler = StandardScaler()
                    normalized_data = scaler.fit_transform(numeric_data)
                    self.data[numeric_data.columns] = normalized_data
                    
                    # Update the table
                    self.update_table(self.normalization_table, self.data)
                    
                    if self.parent:
                        self.parent.data = self.data
                        self.parent.normalized_data = self.data.values
                    
                    QMessageBox.information(self, "Success", "Data normalized successfully.")
                else:
                    QMessageBox.warning(self, "Warning", "No numeric columns found for normalization.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error normalizing data: {str(e)}")
    
    def update_table(self, table_widget, dataframe):
        table_widget.clear()
        if dataframe is not None and not dataframe.empty:
            table_widget.setRowCount(min(dataframe.shape[0], 100))  # Limit to 100 rows for performance
            table_widget.setColumnCount(dataframe.shape[1])
            table_widget.setHorizontalHeaderLabels(dataframe.columns)
            
            # Only show first 100 rows
            display_df = dataframe.head(100)
            
            for i in range(display_df.shape[0]):
                for j in range(display_df.shape[1]):
                    table_widget.setItem(i, j, QTableWidgetItem(str(display_df.iat[i, j])))
        else:
            table_widget.setRowCount(0)
            table_widget.setColumnCount(0)


class ClusteringSection(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.init_ui()
        self.results = {}
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tabs for different clustering algorithms
        self.tabs = QTabWidget()
        
        # Add tabs for each algorithm
        self.kmeans_tab = QWidget()
        self.kmedoids_tab = QWidget()
        self.agnes_tab = QWidget()
        self.diana_tab = QWidget()
        self.dbscan_tab = QWidget()
        self.comparison_tab = QWidget()
        self.results_tab = QWidget()
        
        self.tabs.addTab(self.kmeans_tab, "K-Means")
        self.tabs.addTab(self.kmedoids_tab, "K-Medoids")
        self.tabs.addTab(self.agnes_tab, "AGNES")
        self.tabs.addTab(self.diana_tab, "DIANA")
        self.tabs.addTab(self.dbscan_tab, "DBSCAN")
        self.tabs.addTab(self.comparison_tab, "Compare")
        self.tabs.addTab(self.results_tab, "Results")
        
        # Initialize each tab
        self.init_kmeans_tab()
        self.init_kmedoids_tab()
        self.init_agnes_tab()
        self.init_diana_tab()
        self.init_dbscan_tab()
        self.init_comparison_tab()
        self.init_results_tab()
        
        layout.addWidget(self.tabs)
    
    def init_kmeans_tab(self):
        layout = QVBoxLayout(self.kmeans_tab)
        
        # Parameters section
        params_group = QGroupBox("K-Means Parameters")
        params_layout = QGridLayout(params_group)
        
        # Number of clusters
        n_clusters_label = QLabel("Number of clusters:")
        self.kmeans_n_clusters = QSpinBox()
        self.kmeans_n_clusters.setMinimum(2)
        self.kmeans_n_clusters.setMaximum(20)
        self.kmeans_n_clusters.setValue(3)
        
        # Find optimal button
        self.kmeans_find_optimal = QPushButton("Find Optimal k")
        self.kmeans_find_optimal.clicked.connect(lambda: self.find_optimal_clusters('kmeans'))
        
        # Visual checkbox
        self.kmeans_visualize = QCheckBox("Show visualization")
        self.kmeans_visualize.setChecked(True)
        
        # Add widgets to params layout
        params_layout.addWidget(n_clusters_label, 0, 0)
        params_layout.addWidget(self.kmeans_n_clusters, 0, 1)
        params_layout.addWidget(self.kmeans_find_optimal, 0, 2)
        params_layout.addWidget(self.kmeans_visualize, 1, 0, 1, 2)
        
        # Run button
        run_button = QPushButton("Run K-Means")
        run_button.clicked.connect(self.run_kmeans)
        
        # Add widgets to main layout
        layout.addWidget(params_group)
        layout.addWidget(run_button)
        layout.addStretch()
    
    def init_kmedoids_tab(self):
        layout = QVBoxLayout(self.kmedoids_tab)
        
        # Parameters section
        params_group = QGroupBox("K-Medoids Parameters")
        params_layout = QGridLayout(params_group)
        
        # Number of clusters
        n_clusters_label = QLabel("Number of clusters:")
        self.kmedoids_n_clusters = QSpinBox()
        self.kmedoids_n_clusters.setMinimum(2)
        self.kmedoids_n_clusters.setMaximum(20)
        self.kmedoids_n_clusters.setValue(3)
        
        # Max iterations
        max_iter_label = QLabel("Max iterations:")
        self.kmedoids_max_iter = QSpinBox()
        self.kmedoids_max_iter.setMinimum(1)
        self.kmedoids_max_iter.setMaximum(1000)
        self.kmedoids_max_iter.setValue(300)
        
        # Find optimal button
        self.kmedoids_find_optimal = QPushButton("Find Optimal k")
        self.kmedoids_find_optimal.clicked.connect(lambda: self.find_optimal_clusters('kmedoids'))
        
        # Visual checkbox
        self.kmedoids_visualize = QCheckBox("Show visualization")
        self.kmedoids_visualize.setChecked(True)
        
        # Add widgets to params layout
        params_layout.addWidget(n_clusters_label, 0, 0)
        params_layout.addWidget(self.kmedoids_n_clusters, 0, 1)
        params_layout.addWidget(self.kmedoids_find_optimal, 0, 2)
        params_layout.addWidget(max_iter_label, 1, 0)
        params_layout.addWidget(self.kmedoids_max_iter, 1, 1)
        params_layout.addWidget(self.kmedoids_visualize, 2, 0, 1, 2)
        
        # Run button
        run_button = QPushButton("Run K-Medoids")
        run_button.clicked.connect(self.run_kmedoids)
        
        # Add widgets to main layout
        layout.addWidget(params_group)
        layout.addWidget(run_button)
        layout.addStretch()
    
    def init_agnes_tab(self):
        layout = QVBoxLayout(self.agnes_tab)
        
        # Parameters section
        params_group = QGroupBox("AGNES Parameters")
        params_layout = QGridLayout(params_group)
        
        # Number of clusters
        n_clusters_label = QLabel("Number of clusters:")
        self.agnes_n_clusters = QSpinBox()
        self.agnes_n_clusters.setMinimum(2)
        self.agnes_n_clusters.setMaximum(20)
        self.agnes_n_clusters.setValue(3)
        
        # Linkage method
        linkage_label = QLabel("Linkage method:")
        self.agnes_linkage = QComboBox()
        self.agnes_linkage.addItems(['ward', 'complete', 'average', 'single'])
        
        # Find optimal button
        self.agnes_find_optimal = QPushButton("Find Optimal k")
        self.agnes_find_optimal.clicked.connect(lambda: self.find_optimal_clusters('agnes'))
        
        # Checkboxes
        self.agnes_visualize = QCheckBox("Show visualization")
        self.agnes_visualize.setChecked(True)
        
        self.agnes_dendrogram = QCheckBox("Show dendrogram")
        self.agnes_dendrogram.setChecked(False)
        
        # Add widgets to params layout
        params_layout.addWidget(n_clusters_label, 0, 0)
        params_layout.addWidget(self.agnes_n_clusters, 0, 1)
        params_layout.addWidget(self.agnes_find_optimal, 0, 2)
        params_layout.addWidget(linkage_label, 1, 0)
        params_layout.addWidget(self.agnes_linkage, 1, 1)
        params_layout.addWidget(self.agnes_visualize, 2, 0)
        params_layout.addWidget(self.agnes_dendrogram, 2, 1)
        
        # Run button
        run_button = QPushButton("Run AGNES")
        run_button.clicked.connect(self.run_agnes)
        
        # Add widgets to main layout
        layout.addWidget(params_group)
        layout.addWidget(run_button)
        layout.addStretch()
    
    def init_diana_tab(self):
        layout = QVBoxLayout(self.diana_tab)
        
        # Parameters section
        params_group = QGroupBox("DIANA Parameters")
        params_layout = QGridLayout(params_group)
        
        # Number of clusters
        n_clusters_label = QLabel("Number of clusters:")
        self.diana_n_clusters = QSpinBox()
        self.diana_n_clusters.setMinimum(2)
        self.diana_n_clusters.setMaximum(20)
        self.diana_n_clusters.setValue(3)
        
        # Find optimal button
        self.diana_find_optimal = QPushButton("Find Optimal k")
        self.diana_find_optimal.clicked.connect(lambda: self.find_optimal_clusters('diana'))
        
        # Checkboxes
        self.diana_visualize = QCheckBox("Show visualization")
        self.diana_visualize.setChecked(True)
        
        self.diana_dendrogram = QCheckBox("Show dendrogram")
        self.diana_dendrogram.setChecked(False)
        
        # Add widgets to params layout
        params_layout.addWidget(n_clusters_label, 0, 0)
        params_layout.addWidget(self.diana_n_clusters, 0, 1)
        params_layout.addWidget(self.diana_find_optimal, 0, 2)
        params_layout.addWidget(self.diana_visualize, 1, 0)
        params_layout.addWidget(self.diana_dendrogram, 1, 1)
        
        # Run button
        run_button = QPushButton("Run DIANA")
        run_button.clicked.connect(self.run_diana)
        
        # Add widgets to main layout
        layout.addWidget(params_group)
        layout.addWidget(run_button)
        layout.addStretch()
    
    def init_dbscan_tab(self):
        layout = QVBoxLayout(self.dbscan_tab)
        
        # Parameters section
        params_group = QGroupBox("DBSCAN Parameters")
        params_layout = QGridLayout(params_group)
        
        # Epsilon
        eps_label = QLabel("Epsilon:")
        self.dbscan_eps = QDoubleSpinBox()
        self.dbscan_eps.setMinimum(0.01)
        self.dbscan_eps.setMaximum(10.0)
        self.dbscan_eps.setValue(0.5)
        self.dbscan_eps.setSingleStep(0.1)
        
        # Min samples
        min_samples_label = QLabel("Min samples:")
        self.dbscan_min_samples = QSpinBox()
        self.dbscan_min_samples.setMinimum(1)
        self.dbscan_min_samples.setMaximum(100)
        self.dbscan_min_samples.setValue(5)
        
        # Visual checkbox
        self.dbscan_visualize = QCheckBox("Show visualization")
        self.dbscan_visualize.setChecked(True)
        
        # Add widgets to params layout
        params_layout.addWidget(eps_label, 0, 0)
        params_layout.addWidget(self.dbscan_eps, 0, 1)
        params_layout.addWidget(min_samples_label, 1, 0)
        params_layout.addWidget(self.dbscan_min_samples, 1, 1)
        params_layout.addWidget(self.dbscan_visualize, 2, 0, 1, 2)
        
        # Run button
        run_button = QPushButton("Run DBSCAN")
        run_button.clicked.connect(self.run_dbscan)
        
        # Add widgets to main layout
        layout.addWidget(params_group)
        layout.addWidget(run_button)
        layout.addStretch()
    
    def init_comparison_tab(self):
        layout = QVBoxLayout(self.comparison_tab)
        
        # Parameters section
        params_group = QGroupBox("Comparison Parameters")
        params_layout = QGridLayout(params_group)
        
        # Default number of clusters
        default_k_label = QLabel("Default clusters:")
        self.default_n_clusters = QSpinBox()
        self.default_n_clusters.setMinimum(2)
        self.default_n_clusters.setMaximum(20)
        self.default_n_clusters.setValue(3)
        
        # Individual k values for each algorithm
        kmeans_k_label = QLabel("K-Means clusters:")
        self.comp_kmeans_n_clusters = QSpinBox()
        self.comp_kmeans_n_clusters.setMinimum(0)
        self.comp_kmeans_n_clusters.setMaximum(20)
        self.comp_kmeans_n_clusters.setValue(0)
        self.comp_kmeans_n_clusters.setSpecialValueText("Default")
        
        kmedoids_k_label = QLabel("K-Medoids clusters:")
        self.comp_kmedoids_n_clusters = QSpinBox()
        self.comp_kmedoids_n_clusters.setMinimum(0)
        self.comp_kmedoids_n_clusters.setMaximum(20)
        self.comp_kmedoids_n_clusters.setValue(0)
        self.comp_kmedoids_n_clusters.setSpecialValueText("Default")
        
        agnes_k_label = QLabel("AGNES clusters:")
        self.comp_agnes_n_clusters = QSpinBox()
        self.comp_agnes_n_clusters.setMinimum(0)
        self.comp_agnes_n_clusters.setMaximum(20)
        self.comp_agnes_n_clusters.setValue(0)
        self.comp_agnes_n_clusters.setSpecialValueText("Default")
        
        diana_k_label = QLabel("DIANA clusters:")
        self.comp_diana_n_clusters = QSpinBox()
        self.comp_diana_n_clusters.setMinimum(0)
        self.comp_diana_n_clusters.setMaximum(20)
        self.comp_diana_n_clusters.setValue(0)
        self.comp_diana_n_clusters.setSpecialValueText("Default")
        
        # Add widgets to layout
        params_layout.addWidget(default_k_label, 0, 0)
        params_layout.addWidget(self.default_n_clusters, 0, 1)
        params_layout.addWidget(kmeans_k_label, 1, 0)
        params_layout.addWidget(self.comp_kmeans_n_clusters, 1, 1)
        params_layout.addWidget(kmedoids_k_label, 2, 0)
        params_layout.addWidget(self.comp_kmedoids_n_clusters, 2, 1)
        params_layout.addWidget(agnes_k_label, 3, 0)
        params_layout.addWidget(self.comp_agnes_n_clusters, 3, 1)
        params_layout.addWidget(diana_k_label, 4, 0)
        params_layout.addWidget(self.comp_diana_n_clusters, 4, 1)
        
        # Run button
        run_button = QPushButton("Compare All Algorithms")
        run_button.clicked.connect(self.compare_algorithms)
        
        # Add widgets to main layout
        layout.addWidget(params_group)
        layout.addWidget(run_button)
        layout.addStretch()
    
    def init_results_tab(self):
        layout = QVBoxLayout(self.results_tab)

        # Create a horizontal splitter to have text on left and plot on right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Create text area for results with scrolling
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumWidth(300)

        # Create canvas for plotting
        self.plot_canvas = MatplotlibCanvas()
        self.plot_canvas.setMinimumWidth(400)

        # Add widgets to splitter (text on left, plot on right)
        splitter.addWidget(self.results_text)
        splitter.addWidget(self.plot_canvas)

        # Set initial sizes (roughly 1:1 ratio)
        splitter.setSizes([500, 500])

        # Add the splitter to the layout
        layout.addWidget(splitter)
    
    def find_optimal_clusters(self, algorithm):
        if self.parent and self.parent.normalized_data is not None:
            try:
                # Show message that this might take time
                QMessageBox.information(self, "Please Wait", 
                    f"Finding optimal number of clusters for {algorithm.upper()}.\nThis may take a moment...")
                
                # Call method from parent (MLApp) to find optimal clusters
                optimal_k = self.parent.find_optimal_clusters(
                    method='elbow', 
                    max_clusters=10, 
                    algorithm=algorithm
                )
                
                # Update the appropriate spinner with the optimal k
                if algorithm == 'kmeans':
                    self.kmeans_n_clusters.setValue(optimal_k)
                elif algorithm == 'kmedoids':
                    self.kmedoids_n_clusters.setValue(optimal_k)
                elif algorithm == 'agnes':
                    self.agnes_n_clusters.setValue(optimal_k)
                elif algorithm == 'diana':
                    self.diana_n_clusters.setValue(optimal_k)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error finding optimal clusters: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load and preprocess data first!")
    
    def run_kmeans(self):
        if self.parent and self.parent.normalized_data is not None:
            try:
                n_clusters = self.kmeans_n_clusters.value()
                visualize = self.kmeans_visualize.isChecked()
                
                # Call run_kmeans method from parent
                result = self.parent.run_kmeans(n_clusters=n_clusters, visualize=visualize)
                
                if result:
                    # Store for comparison
                    self.results['kmeans'] = result
                    
                    # Display results
                    self.display_results(f"K-Means Clustering (k={n_clusters})", result)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error running K-Means: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load and preprocess data first!")
    
    def run_kmedoids(self):
        if self.parent and self.parent.normalized_data is not None:
            try:
                n_clusters = self.kmedoids_n_clusters.value()
                max_iter = self.kmedoids_max_iter.value()
                visualize = self.kmedoids_visualize.isChecked()
                
                # Call run_kmedoids method from parent
                result = self.parent.run_kmedoids(
                    n_clusters=n_clusters, 
                    max_iter=max_iter, 
                    visualize=visualize
                )
                
                if result:
                    # Store for comparison
                    self.results['kmedoids'] = result
                    
                    # Display results
                    self.display_results(f"K-Medoids Clustering (k={n_clusters})", result)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error running K-Medoids: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load and preprocess data first!")
    
    def run_agnes(self):
        if self.parent and self.parent.normalized_data is not None:
            try:
                n_clusters = self.agnes_n_clusters.value()
                linkage_method = self.agnes_linkage.currentText()
                visualize = self.agnes_visualize.isChecked()
<<<<<<< HEAD
                plot_dendrogram = self.agnes_dendrogram.isChecked()
                
                # Call run_agnes method from parent
                result = self.parent.run_agnes(
                    n_clusters=n_clusters,
                    linkage=linkage_method,
                    visualize=visualize,
                    plot_dendrogram=plot_dendrogram
=======
                show_dendrogram = self.agnes_dendrogram.isChecked()

                # Call run_agnes method from parent
                result = self.parent.run_agnes(
                    n_clusters=n_clusters,
                    linkage_method=linkage_method,
                    visualize=visualize,
                    show_dendrogram=show_dendrogram
>>>>>>> c261d8f72717591b2d1a421c5fa94976f9e6dc28
                )
                
                if result:
                    # Store for comparison
                    self.results['agnes'] = result
                    
                    # Display results
                    self.display_results(f"AGNES Clustering (k={n_clusters}, linkage={linkage_method})", result)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error running AGNES: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load and preprocess data first!")
<<<<<<< HEAD
    
=======

>>>>>>> c261d8f72717591b2d1a421c5fa94976f9e6dc28
    def run_diana(self):
        if self.parent and self.parent.normalized_data is not None:
            try:
                n_clusters = self.diana_n_clusters.value()
                visualize = self.diana_visualize.isChecked()
                plot_dendrogram = self.diana_dendrogram.isChecked()
                
                # Call run_diana method from parent
                result = self.parent.run_diana(
                    n_clusters=n_clusters,
                    visualize=visualize,
                    plot_dendrogram=plot_dendrogram
                )
                
                if result:
                    # Store for comparison
                    self.results['diana'] = result
                    
                    # Display results
                    self.display_results(f"DIANA Clustering (k={n_clusters})", result)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error running DIANA: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load and preprocess data first!")
    
    def run_dbscan(self):
        if self.parent and self.parent.normalized_data is not None:
            try:
                eps = self.dbscan_eps.value()
                min_samples = self.dbscan_min_samples.value()
                visualize = self.dbscan_visualize.isChecked()
                
                # Call run_dbscan method from parent
                result = self.parent.run_dbscan(
                    eps=eps,
                    min_samples=min_samples,
                    visualize=visualize
                )
                
                if result:
                    # Store for comparison
                    self.results['dbscan'] = result
                    
                    # Display results
                    self.display_results(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})", result)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error running DBSCAN: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load and preprocess data first!")
    
    def compare_algorithms(self):
        if self.parent and self.parent.normalized_data is not None:
            try:
                # Get parameters
                default_k = self.default_n_clusters.value()
                
                # Get individual k values or use default
                kmeans_k = self.comp_kmeans_n_clusters.value() if self.comp_kmeans_n_clusters.value() > 0 else default_k
                kmedoids_k = self.comp_kmedoids_n_clusters.value() if self.comp_kmedoids_n_clusters.value() > 0 else default_k
                agnes_k = self.comp_agnes_n_clusters.value() if self.comp_agnes_n_clusters.value() > 0 else default_k
                diana_k = self.comp_diana_n_clusters.value() if self.comp_diana_n_clusters.value() > 0 else default_k
                
                # Run comparison
                comparison_df = self.parent.compare_algorithms(
                    n_clusters=default_k,
                    n_clusters_kmeans=kmeans_k,
                    n_clusters_kmedoids=kmedoids_k,
                    n_clusters_agnes=agnes_k,
                    n_clusters_diana=diana_k
                )
                
                # Display results
                self.display_comparison_results(comparison_df)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error comparing algorithms: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load and preprocess data first!")
    
    def display_results(self, title, result):
        if result:
            # Switch to results tab
            self.tabs.setCurrentWidget(self.results_tab)
            
            # Create text summary
            result_text = f"=== {title} ===\n\n"
            result_text += f"Number of clusters: {result['metrics']['n_clusters']}\n"
            result_text += f"Silhouette score: {result['metrics']['silhouette_score']:.4f}\n"
            result_text += f"Calinski-Harabasz index: {result['metrics']['calinski_harabasz']:.4f}\n"
            result_text += f"Davies-Bouldin index: {result['metrics']['davies_bouldin']:.4f}\n"
            result_text += f"Intra-class distance: {result['metrics']['intra_class_distance']:.4f}\n"
            result_text += f"Inter-class distance: {result['metrics']['inter_class_distance']:.4f}\n"
            result_text += f"Execution time: {result['metrics']['execution_time']:.4f} seconds\n"
            
            # Add cluster sizes if available
            if 'labels' in result:
                unique, counts = np.unique(result['labels'], return_counts=True)
                result_text += "\nCluster sizes:\n"
                for cluster, size in zip(unique, counts):
                    result_text += f"  Cluster {cluster}: {size} samples\n"
            
            # Display the text
            self.results_text.setText(result_text)
            
            # Clear previous plot and create new one
            self.plot_canvas.axes.clear()
            
            # Visualize clusters using PCA
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(self.parent.normalized_data)
            
            # Plot clusters
            scatter = self.plot_canvas.axes.scatter(
                reduced_data[:, 0], 
                reduced_data[:, 1], 
                c=result['labels'], 
                cmap='viridis',
                alpha=0.7
            )
            
            # Add plot details
            self.plot_canvas.axes.set_title(f"{title} (PCA-reduced)")
            self.plot_canvas.axes.set_xlabel('Principal Component 1')
            self.plot_canvas.axes.set_ylabel('Principal Component 2')
            self.plot_canvas.figure.colorbar(scatter, ax=self.plot_canvas.axes, label='Cluster')
            self.plot_canvas.axes.grid(True)
            
            # Redraw the canvas
            self.plot_canvas.draw()
    
    def display_comparison_results(self, comparison_df):
        if comparison_df is not None and not comparison_df.empty:
            # Switch to results tab
            self.tabs.setCurrentWidget(self.results_tab)
            
            # Create text summary
            result_text = "=== Algorithm Comparison ===\n\n"
            result_text += comparison_df.to_string()
            
            # Display the text
            self.results_text.setText(result_text)
            
            # Clear previous plot and create new comparison plots
            self.plot_canvas.axes.clear()
            
            # Plot comparison metrics
            metrics_to_plot = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
            
            # Create subplots
            fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
            
            for i, metric in enumerate(metrics_to_plot):
                axes[i].bar(comparison_df.index, comparison_df[metric])
                axes[i].set_title(metric)
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
            
            # Display the plot
            self.plot_canvas.figure = fig
            self.plot_canvas.draw()
        else:
            self.results_text.setText("No valid comparison results available.")


class ClusteringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Data attributes
        self.data = None
        self.raw_data = None
        self.normalized_data = None
        self.label_encoders = {}
        self.results = {}
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        # Main application setup
        self.setWindowTitle("Clustering Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create tabs for different sections
        tabs = QTabWidget()
        
        # Create preprocessing section
        self.preprocess_section = UploadAndPreprocess(self)
        tabs.addTab(self.preprocess_section, "Data Preprocessing")
        
        # Create clustering section
        self.clustering_section = ClusteringSection(self)
        tabs.addTab(self.clustering_section, "Clustering")
        
        main_layout.addWidget(tabs)
    
    def update_feature_selectors(self):
        # Update any feature selectors with column names
        pass
    
    def find_optimal_clusters(self, method='elbow', max_clusters=10, algorithm='kmeans'):
        """Find the optimal number of clusters using the elbow method"""
        if self.normalized_data is None:
            return None
        
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
        """Run K-Means clustering algorithm"""
        if self.normalized_data is None:
            return None
            
        try:
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
            
            result = {
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
            
            self.results['kmeans'] = result
            
            if visualize:
                self._visualize_clusters(cluster_labels, algorithm='K-Means')
            
            return result
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error running K-Means: {str(e)}")
            return None
    
    def run_kmedoids(self, n_clusters=3, max_iter=300, random_state=None, visualize=True):
        """Run K-medoids clustering algorithm"""
        if self.normalized_data is None:
            return None
            
        try:
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
<<<<<<< HEAD
                    # Get all non-medoid points in this cluster
                    cluster_points = np.where(labels == i)[0]
                    non_medoids = np.setdiff1d(cluster_points, medoid_indices)
=======
                    cluster_points = X[labels == i]
                    if len(cluster_points) > 0:
                        intra_distances = pairwise_distances(cluster_points, metric='euclidean')
                        min_idx = np.argmin(np.sum(intra_distances, axis=1))
                        new_medoids[i] = cluster_points[min_idx]
                
                # Check convergence
                if np.array_equal(medoids, new_medoids):
                    break
                
                medoids = new_medoids
            
            return labels, medoids
        
        def run_kmeans(self, n_clusters=3, visualize=True):
            """Run K-means clustering algorithm"""
            if self.normalized_data is None:
                return None
            
            result = {}
            start_time = timer()
            
            # Run K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.normalized_data)
            
            # Calculate execution time
            execution_time = timer() - start_time
            
            # Store results
            result['n_clusters'] = n_clusters
            result['labels'] = labels
            result['cluster_sizes'] = np.bincount(labels)
            result['time'] = execution_time
            
            # Calculate metrics
            if len(np.unique(labels)) > 1:
                result['silhouette'] = silhouette_score(self.normalized_data, labels)
                result['calinski_harabasz'] = calinski_harabasz_score(self.normalized_data, labels)
                result['davies_bouldin'] = davies_bouldin_score(self.normalized_data, labels)
            else:
                result['silhouette'] = float('nan')
                result['calinski_harabasz'] = float('nan')
                result['davies_bouldin'] = float('nan')
            
            if visualize:
                # Visualize using PCA if needed
                if self.normalized_data.shape[1] > 2:
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(self.normalized_data)
                    explained_variance = pca.explained_variance_ratio_
                else:
                    reduced_data = self.normalized_data
                    explained_variance = [1.0, 1.0] if self.normalized_data.shape[1] == 2 else [1.0]
                
                # Create visualization
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
                plt.title(f'K-means Clustering (k={n_clusters})')
                plt.xlabel(f'Component 1 ({explained_variance[0]:.2%} variance)')
                if reduced_data.shape[1] > 1:
                    plt.ylabel(f'Component 2 ({explained_variance[1]:.2%} variance)')
                plt.colorbar(label='Cluster')
                plt.grid(True)
                
                result['figure'] = fig
            
            return result
        
        def run_kmedoids(self, n_clusters=3, max_iter=300, visualize=True):
            """Run K-medoids clustering algorithm"""
            if self.normalized_data is None:
                return None
            
            result = {}
            start_time = timer()
            
            # Run K-medoids
            labels, medoids = self._kmedoids(self.normalized_data, n_clusters, max_iter)
            
            # Calculate execution time
            execution_time = timer() - start_time
            
            # Store results
            result['n_clusters'] = n_clusters
            result['labels'] = labels
            result['cluster_centers'] = medoids
            result['cluster_sizes'] = np.bincount(labels)
            result['time'] = execution_time
            
            # Calculate metrics
            if len(np.unique(labels)) > 1:
                result['silhouette'] = silhouette_score(self.normalized_data, labels)
                result['calinski_harabasz'] = calinski_harabasz_score(self.normalized_data, labels)
                result['davies_bouldin'] = davies_bouldin_score(self.normalized_data, labels)
            else:
                result['silhouette'] = float('nan')
                result['calinski_harabasz'] = float('nan')
                result['davies_bouldin'] = float('nan')
            
            if visualize:
                # Visualization code similar to kmeans
                if self.normalized_data.shape[1] > 2:
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(self.normalized_data)
                    reduced_medoids = pca.transform(medoids)
                    explained_variance = pca.explained_variance_ratio_
                else:
                    reduced_data = self.normalized_data
                    reduced_medoids = medoids
                    explained_variance = [1.0, 1.0] if self.normalized_data.shape[1] == 2 else [1.0]
                
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
                plt.scatter(reduced_medoids[:, 0], reduced_medoids[:, 1], c='red', marker='x', s=200, alpha=1)
                plt.title(f'K-medoids Clustering (k={n_clusters})')
                plt.xlabel(f'Component 1 ({explained_variance[0]:.2%} variance)')
                if reduced_data.shape[1] > 1:
                    plt.ylabel(f'Component 2 ({explained_variance[1]:.2%} variance)')
                plt.colorbar(label='Cluster')
                plt.grid(True)
                
                result['figure'] = fig
            
            return result

        def run_dbscan(self, eps=0.5, min_samples=5, visualize=True):
            """Run DBSCAN clustering algorithm"""
            if self.normalized_data is None:
                return None
            
            result = {}
            start_time = timer()
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(self.normalized_data)
            
            # Calculate execution time
            execution_time = timer() - start_time
            
            # Calculate metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            result['n_clusters'] = n_clusters
            result['labels'] = labels
            result['cluster_sizes'] = np.bincount(labels[labels >= 0])
            result['time'] = execution_time
            result['noise_points'] = np.sum(labels == -1)
            
            # Calculate metrics if possible
            if n_clusters > 1:
                valid_indices = labels != -1
                if np.sum(valid_indices) > 1:
                    result['silhouette'] = silhouette_score(
                        self.normalized_data[valid_indices], 
                        labels[valid_indices]
                    )
                    result['calinski_harabasz'] = calinski_harabasz_score(
                        self.normalized_data[valid_indices], 
                        labels[valid_indices]
                    )
                    result['davies_bouldin'] = davies_bouldin_score(
                        self.normalized_data[valid_indices], 
                        labels[valid_indices]
                    )
                else:
                    result['silhouette'] = float('nan')
                    result['calinski_harabasz'] = float('nan')
                    result['davies_bouldin'] = float('nan')
            else:
                result['silhouette'] = float('nan')
                result['calinski_harabasz'] = float('nan')
                result['davies_bouldin'] = float('nan')
            
            if visualize:
                # Create visualization
                fig = plt.figure(figsize=(10, 6))
                
                if self.normalized_data.shape[1] > 2:
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(self.normalized_data)
                    explained_variance = pca.explained_variance_ratio_
                else:
                    reduced_data = self.normalized_data
                    explained_variance = [1.0, 1.0] if self.normalized_data.shape[1] == 2 else [1.0]
                
                # Plot with special handling for noise points
                unique_labels = set(labels)
                colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
                
                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        col = [0, 0, 0, 1]  # Black for noise
>>>>>>> c261d8f72717591b2d1a421c5fa94976f9e6dc28
                    
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
            intra_class = self._calculate_intra_class_distance(X, labels)
            inter_class = self._calculate_inter_class_distance(X, labels)

            end_time = timer()
            execution_time = end_time - start_time
            
            result = {
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
            
            self.results['kmedoids'] = result
            
            if visualize:
                self._visualize_clusters(labels, algorithm='K-Medoids')
            
            return result
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error running K-Medoids: {str(e)}")
            return None
    
    def run_agnes(self, n_clusters=3, linkage='ward', visualize=True, plot_dendrogram=False):
        """Run Agglomerative Hierarchical Clustering (AGNES)"""
        if self.normalized_data is None:
            return None
            
        try:
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
            
            result = {
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
            
            self.results['agnes'] = result
            
            if visualize:
                self._visualize_clusters(cluster_labels, algorithm='AGNES')
                
            if plot_dendrogram:
                self._plot_dendrogram()
            
            return result
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error running AGNES: {str(e)}")
            return None
    
    def run_diana(self, n_clusters=3, visualize=True, plot_dendrogram=False):
        """Run Divisive Hierarchical Clustering (DIANA)"""
        if self.normalized_data is None:
            return None
            
        try:
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
            
            result = {
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
            
            self.results['diana'] = result
            
            if visualize:
                self._visualize_clusters(labels, algorithm='DIANA')
                
            if plot_dendrogram:
                self._plot_dendrogram()
            
            return result
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error running DIANA: {str(e)}")
            return None
    
    def run_dbscan(self, eps=0.5, min_samples=5, visualize=True):
        """Run DBSCAN clustering algorithm"""
        if self.normalized_data is None:
            return None
            
        try:
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
            
            result = {
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
            
            self.results['dbscan'] = result
            
            if visualize:
                self._visualize_clusters(cluster_labels, algorithm='DBSCAN')
            
            return result
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error running DBSCAN: {str(e)}")
            return None
    
    def _plot_dendrogram(self, **kwargs):
        """Plot hierarchical clustering dendrogram"""
        plt.figure(figsize=(10, 7))
        plt.title("Hierarchical Clustering Dendrogram")
        
        Z = linkage(self.normalized_data, 'ward')
        dendrogram(Z, **kwargs)
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        plt.show()
    
    def _visualize_clusters(self, labels, algorithm=''):
        """Visualize clusters using PCA for dimensionality reduction"""
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
    
    def compare_algorithms(self, n_clusters=3, n_clusters_diana=None, n_clusters_kmeans=None,
                         n_clusters_kmedoids=None, n_clusters_agnes=None):
        """Compare all clustering algorithms"""
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
        
        # Create and display comparison table
        df = pd.DataFrame(metrics)
        if df.empty:
            print("Warning: No valid clustering results to compare")
            return df
        
        df.set_index('Algorithm', inplace=True)
        
        return df

        def run_agnes(self, n_clusters=3, linkage_method='ward', visualize=True, show_dendrogram=False):
            """Run Agglomerative Hierarchical Clustering (AGNES)"""
            if self.normalized_data is None:
                return None
            
            result = {}
            start_time = timer()
            
            # Run AGNES
            agnes = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            labels = agnes.fit_predict(self.normalized_data)
            
            # Calculate execution time
            execution_time = timer() - start_time
            
            # Store results
            result['n_clusters'] = n_clusters
            result['labels'] = labels
            result['cluster_sizes'] = np.bincount(labels)
            result['time'] = execution_time
            
            # Calculate metrics
            if len(np.unique(labels)) > 1:
                result['silhouette'] = silhouette_score(self.normalized_data, labels)
                result['calinski_harabasz'] = calinski_harabasz_score(self.normalized_data, labels)
                result['davies_bouldin'] = davies_bouldin_score(self.normalized_data, labels)
            else:
                result['silhouette'] = float('nan')
                result['calinski_harabasz'] = float('nan')
                result['davies_bouldin'] = float('nan')
            
            # Create visualization
            if visualize:
                fig = plt.figure(figsize=(10, 6))
                
                if self.normalized_data.shape[1] > 2:
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(self.normalized_data)
                    explained_variance = pca.explained_variance_ratio_
                else:
                    reduced_data = self.normalized_data
                    explained_variance = [1.0, 1.0] if self.normalized_data.shape[1] == 2 else [1.0]
                
                plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
                plt.title(f'AGNES Clustering (k={n_clusters}, linkage={linkage_method})')
                plt.xlabel(f'Component 1 ({explained_variance[0]:.2%} variance)')
                if reduced_data.shape[1] > 1:
                    plt.ylabel(f'Component 2 ({explained_variance[1]:.2%} variance)')
                plt.colorbar(label='Cluster')
                plt.grid(True)
                
                result['figure'] = fig
            
            # Show dendrogram in a separate window if requested
            if show_dendrogram:
                plt.figure(figsize=(12, 8))
                Z = linkage(self.normalized_data, method=linkage_method)
                dendrogram(Z)
                plt.title(f'AGNES Hierarchical Clustering Dendrogram (linkage={linkage_method})')
                plt.xlabel('Sample index')
                plt.ylabel('Distance')
                plt.tight_layout()
                plt.show()
            
            return result
        
        def run_diana(self, n_clusters=3, visualize=True, show_dendrogram=False):
            """Run Divisive Hierarchical Clustering (DIANA)"""
            if self.normalized_data is None:
                return None
            
            result = {}
            start_time = timer()
            
            # DIANA implementation using AgglomerativeClustering as approximation
            # since sklearn doesn't provide DIANA
            diana = AgglomerativeClustering(n_clusters=n_clusters)
            labels = diana.fit_predict(self.normalized_data)
            
            # Calculate execution time
            execution_time = timer() - start_time
            
            # Store results
            result['n_clusters'] = n_clusters
            result['labels'] = labels
            result['cluster_sizes'] = np.bincount(labels)
            result['time'] = execution_time
            
            # Calculate metrics
            if len(np.unique(labels)) > 1:
                result['silhouette'] = silhouette_score(self.normalized_data, labels)
                result['calinski_harabasz'] = calinski_harabasz_score(self.normalized_data, labels)
                result['davies_bouldin'] = davies_bouldin_score(self.normalized_data, labels)
            else:
                result['silhouette'] = float('nan')
                result['calinski_harabasz'] = float('nan')
                result['davies_bouldin'] = float('nan')
            
            if visualize:
                # Create visualization
                fig = plt.figure(figsize=(10, 6))
                
                if self.normalized_data.shape[1] > 2:
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(self.normalized_data)
                    explained_variance = pca.explained_variance_ratio_
                else:
                    reduced_data = self.normalized_data
                    explained_variance = [1.0, 1.0] if self.normalized_data.shape[1] == 2 else [1.0]
                
                plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
                plt.title(f'DIANA Clustering (k={n_clusters})')
                plt.xlabel(f'Component 1 ({explained_variance[0]:.2%} variance)')
                if reduced_data.shape[1] > 1:
                    plt.ylabel(f'Component 2 ({explained_variance[1]:.2%} variance)')
                plt.colorbar(label='Cluster')
                plt.grid(True)
                
                result['figure'] = fig
            
            # Show dendrogram in a separate window if requested
            if show_dendrogram:
                plt.figure(figsize=(12, 8))
                # For DIANA, we approximate using complete linkage
                # as it tends to separate larger clusters first
                Z = linkage(self.normalized_data, method='complete')
                dendrogram(Z)
                plt.title('DIANA Hierarchical Clustering Dendrogram')
                plt.xlabel('Sample index')
                plt.ylabel('Distance')
                plt.tight_layout()
                plt.show()
            
            return result


def main():
    app = QApplication(sys.argv)
    clustering_app = ClusteringApp()
    clustering_app.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()