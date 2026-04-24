import json
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def show_confusion_matrix():
    """Display confusion matrix from stored evaluation data."""
    
    print("=" * 60)
    print("DROWSINESS DETECTION - CONFUSION MATRIX")
    print("=" * 60)
    
    try:
        # Load stored data
        with open('evaluation_data.json', 'r') as f:
            data = json.load(f)
        
        samples = data.get('samples', [])
        
        if len(samples) == 0:
            print("\nNo evaluation data found!")
            print("Please run 'python evaluate_model.py' first to collect samples.")
            return
        
        # Reconstruct predictions and ground truth
        predictions = []
        ground_truth = []
        
        for sample in samples:
            predictions.append(1 if sample['predicted'] == 'Drowsy' else 0)
            ground_truth.append(1 if sample['actual'] == 'Drowsy' else 0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        accuracy = accuracy_score(ground_truth, predictions)
        
        print(f"\nTotal Samples: {len(samples)}")
        print(f"Accuracy: {accuracy:.2%}")
        
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("              Awake  Drowsy")
        print(f"Actual Awake    {cm[0][0]:3d}    {cm[0][1]:3d}")
        print(f"       Drowsy   {cm[1][0]:3d}    {cm[1][1]:3d}")
        
        # Calculate detailed metrics
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nDetailed Metrics:")
        print(f"True Positives (TP):  {tp:3d} - Correctly detected drowsy")
        print(f"True Negatives (TN):  {tn:3d} - Correctly detected awake")
        print(f"False Positives (FP): {fp:3d} - False alarm (predicted drowsy, actually awake)")
        print(f"False Negatives (FN): {fn:3d} - Missed drowsy (predicted awake, actually drowsy)")
        
        print(f"\nPerformance Metrics:")
        print(f"Precision:    {precision:6.2%} - When it says drowsy, how often is it right?")
        print(f"Recall:       {recall:6.2%} - How many actual drowsy cases did it catch?")
        print(f"F1-Score:     {f1_score:6.2%} - Overall balance of precision and recall")
        print(f"Specificity:  {specificity:6.2%} - How well does it avoid false alarms?")
        
        # Count by actual state
        awake_count = sum(1 for s in samples if s['actual'] == 'Awake')
        drowsy_count = sum(1 for s in samples if s['actual'] == 'Drowsy')
        
        print(f"\nDataset Distribution:")
        print(f"Awake samples:  {awake_count} ({awake_count/len(samples)*100:.1f}%)")
        print(f"Drowsy samples: {drowsy_count} ({drowsy_count/len(samples)*100:.1f}%)")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Awake', 'Drowsy'],
                    yticklabels=['Awake', 'Drowsy'],
                    cbar_kws={'label': 'Count'})
        
        plt.title(f'Drowsiness Detection Confusion Matrix\n(Accuracy: {accuracy:.2%}, Samples: {len(samples)})', 
                  fontsize=14, fontweight='bold')
        plt.ylabel('Actual State', fontsize=12)
        plt.xlabel('Predicted State', fontsize=12)
        
        # Add metrics text
        metrics_text = f'Precision: {precision:.2%}\nRecall: {recall:.2%}\nF1-Score: {f1_score:.2%}'
        plt.text(2.3, 0.5, metrics_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix plot saved to 'confusion_matrix.png'")
        plt.show()
        
        print("\n" + "=" * 60)
        
    except FileNotFoundError:
        print("\nError: 'evaluation_data.json' not found!")
        print("Please run 'python evaluate_model.py' first to collect evaluation data.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    show_confusion_matrix()
