"""
Data Visualization Utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


class Visualizer:
    def __init__(self, style='seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_training_history(self, history, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_confusion_matrix(self, cm, class_names, save_path=None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_feature_importance(self, features, importances, top_n=20, save_path=None):
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [features[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_roc_curve(self, fpr, tpr, auc_score, save_path=None):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_distribution(self, data, title='Data Distribution', save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(data, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{title} - Histogram')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(data, vert=True)
        ax2.set_ylabel('Value')
        ax2.set_title(f'{title} - Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_correlation_matrix(self, df, save_path=None):
        corr = df.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_interactive_scatter(self, df, x_col, y_col, color_col=None, title='Scatter Plot'):
        if color_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                           title=title, hover_data=df.columns)
        else:
            fig = px.scatter(df, x=x_col, y=y_col, title=title, 
                           hover_data=df.columns)
        
        fig.show()
        
    def plot_interactive_line(self, df, x_col, y_cols, title='Line Plot'):
        fig = go.Figure()
        
        for y_col in y_cols:
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], 
                                    mode='lines', name=y_col))
        
        fig.update_layout(title=title, xaxis_title=x_col, 
                         yaxis_title='Value', hovermode='x unified')
        fig.show()
        
    def plot_3d_scatter(self, df, x_col, y_col, z_col, color_col=None, title='3D Scatter Plot'):
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, 
                           color=color_col, title=title)
        fig.show()


def create_visualizer(style='seaborn'):
    return Visualizer(style=style)