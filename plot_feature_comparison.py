import matplotlib.pyplot as plt
import numpy as np

def plot_feature_comparison(results, title='Feature Set Comparison'):
    """
    Create overlaid plots comparing different feature sets on the same graphs
    
    Args:
        results: Dictionary containing results for each feature set:
                {
                    'All Features': {'history': history_dict, 'metrics': metrics_dict},
                    'Low-Level Features': {'history': history_dict, 'metrics': metrics_dict},
                    'High-Level Features': {'history': history_dict, 'metrics': metrics_dict}
                }
        title: Base title for the plots
        
    Returns:
        None (displays and saves plots)
    """
    # Define colors and line styles for consistent plotting
    colors = {
        'All Features': 'blue',
        'Low-Level Features': 'green',
        'High-Level Features': 'red'
    }
    
    line_styles = {
        'All Features': '-',
        'Low-Level Features': '--',
        'High-Level Features': '-.'
    }
    
    # Create figure for training history
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for feature_set, result in results.items():
        history = result['history']
        plt.plot(
            history['train_loss'], 
            color=colors[feature_set], 
            linestyle=line_styles[feature_set],
            label=f"{feature_set} - Train"
        )
        plt.plot(
            history['val_loss'], 
            color=colors[feature_set], 
            linestyle='--',
            marker='o',
            markersize=4,
            label=f"{feature_set} - Validation"
        )
    
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    for feature_set, result in results.items():
        history = result['history']
        plt.plot(
            history['val_accuracy'], 
            color=colors[feature_set], 
            linestyle=line_styles[feature_set],
            marker='o',
            markersize=4,
            label=f"{feature_set}"
        )
    
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot AUC
    plt.subplot(2, 2, 3)
    for feature_set, result in results.items():
        history = result['history']
        plt.plot(
            history['val_auc'], 
            color=colors[feature_set], 
            linestyle=line_styles[feature_set],
            marker='o',
            markersize=4,
            label=f"{feature_set}"
        )
    
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    # Plot AP
    plt.subplot(2, 2, 4)
    for feature_set, result in results.items():
        history = result['history']
        plt.plot(
            history['val_ap'], 
            color=colors[feature_set], 
            linestyle=line_styles[feature_set],
            marker='o',
            markersize=4,
            label=f"{feature_set}"
        )
    
    plt.title('Validation Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('AP')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(f"{title} - Training History", fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig(f"{title.replace(' ', '_').lower()}_training_history.png")
    plt.show()
    
    # Create figure for ROC and PR curves
    plt.figure(figsize=(15, 6))
    
    # Plot ROC curves
    plt.subplot(1, 2, 1)
    for feature_set, result in results.items():
        metrics = result['metrics']
        fpr, tpr = metrics['roc_curve']
        plt.plot(
            fpr, tpr, 
            color=colors[feature_set], 
            linestyle=line_styles[feature_set],
            label=f"{feature_set} (AUC = {metrics['auc']:.4f})"
        )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    
    # Plot PR curves
    plt.subplot(1, 2, 2)
    for feature_set, result in results.items():
        metrics = result['metrics']
        precision, recall = metrics['pr_curve']
        plt.plot(
            recall, precision, 
            color=colors[feature_set], 
            linestyle=line_styles[feature_set],
            label=f"{feature_set} (AP = {metrics['ap']:.4f})"
        )
    
    plt.title('Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(f"{title} - Evaluation Curves", fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    # Save the figure
    plt.savefig(f"{title.replace(' ', '_').lower()}_evaluation_curves.png")
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Example code - not run when imported
    pass