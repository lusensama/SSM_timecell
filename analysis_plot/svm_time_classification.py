"""
SVM Time Classification Experiment - Lap Counting Data

Experiment design:
- Data: lap_counting_2_activity.npy (5000 episodes, 133 time bins, 80 neurons)
- Extract 4 time slices as classes:
  * Class 1: time steps 5-24 (20 bins)
  * Class 2: time steps 38-57 (20 bins)
  * Class 3: time steps 71-90 (20 bins)
  * Class 4: time steps 105-124 (20 bins)
- Training set: 500 random episodes (10%)
- Test set: 4500 remaining episodes (90%)
- Classifier: SVM
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

load_path = "/home/lugroup/Documents/Sen_Code/time_cell_submit/capsule-0490840-code/data/figure_5_SVM"
plot_dir = "/home/lugroup/Documents/Sen_Code/time_cell_submit/capsule-0490840-code/figures"

def load_data(path):
    """Load data and convert to real-valued."""
    raw = np.load(path, allow_pickle=True)
    
    if np.iscomplexobj(raw):
        data = raw.real
    else:
        data = raw
    
    data = data.astype(np.float64)
    
    return data

def extract_time_slices(activity, use_middle_only=True, middle_size=5):
    """
    Extract 4 time slices; optionally use only the middle time steps.
    
    Args:
        activity: shape (5000, 133, 80)
        use_middle_only: whether to use only the middle time steps
        middle_size: if use_middle_only=True, how many middle time steps to use
    
    Returns:
        slices_list: 4 arrays, each shape (5000, middle_size, 80) or (5000, 20, 80)
        time_ranges: list of original time ranges
        actual_ranges: list of actual time ranges used
    """
    time_ranges = [
        (5, 24),
        (38, 57),
        (71, 90),
        (105, 124)
    ]
    
    slices_list = []
    actual_ranges = []
    
    for start, end in time_ranges:
        total_bins = end - start + 1
        
        if use_middle_only:
            offset = (total_bins - middle_size) // 2
            actual_start = start + offset
            actual_end = actual_start + middle_size - 1
            
            time_slice = activity[:, actual_start:actual_end+1, :]
            actual_ranges.append((actual_start, actual_end))
            
            print(f"  Class {len(slices_list)+1}: original [{start}, {end}] -> middle [{actual_start}, {actual_end}], shape: {time_slice.shape}")
        else:
            time_slice = activity[:, start:end+1, :]
            actual_ranges.append((start, end))
            print(f"  Class {len(slices_list)+1}: time steps [{start}, {end}], shape: {time_slice.shape}")
        
        slices_list.append(time_slice)
    
    return slices_list, time_ranges, actual_ranges

def prepare_dataset(slices_list, test_size=4500, random_seed=42):
    """
    Prepare training and test datasets.
    
    Args:
        slices_list: 4 time slices, each shape (5000, n_time, 80)
        test_size: size of test set
        random_seed: random seed
    
    Returns:
        X_train, X_test, y_train, y_test, train_indices, test_indices
    """
    np.random.seed(random_seed)
    
    n_episodes = slices_list[0].shape[0]
    n_classes = len(slices_list)
    
    indices = np.arange(n_episodes)
    np.random.shuffle(indices)
    
    train_indices = indices[:n_episodes - test_size]
    test_indices = indices[n_episodes - test_size:]
    
    print(f"\nDataset split:")
    print(f"  Training episodes: {len(train_indices)}")
    print(f"  Test episodes: {len(test_indices)}")
    print(f"  Training indices: {train_indices[:10]}... (showing first 10)")
    print(f"  Test indices: {test_indices}")
    
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    
    for class_idx, time_slice in enumerate(slices_list):
        
        train_slice = time_slice[train_indices]
        train_flat = train_slice.reshape(len(train_indices), -1)
        X_train_list.append(train_flat)
        y_train_list.append(np.full(len(train_indices), class_idx))
        
        test_slice = time_slice[test_indices]
        test_flat = test_slice.reshape(len(test_indices), -1)
        X_test_list.append(test_flat)
        y_test_list.append(np.full(len(test_indices), class_idx))
    
    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)
    
    n_time_bins = slices_list[0].shape[1]
    n_neurons = slices_list[0].shape[2]
    
    print(f"\nDataset shapes:")
    print(f"  X_train: {X_train.shape} (samples, features)")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    print(f"  Feature dimensions: {n_time_bins} time bins x {n_neurons} neurons = {n_time_bins*n_neurons}")
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices

def train_svm(X_train, y_train, kernel='rbf', C=1.0):
    """
    Train an SVM classifier.
    
    Args:
        X_train: training features
        y_train: training labels
        kernel: SVM kernel ('linear', 'rbf', 'poly')
        C: regularization parameter
    
    Returns:
        clf: trained classifier
        scaler: standardizer
    """
    print(f"\nTrain SVM classifier:")
    print(f"  Kernel: {kernel}")
    print(f"  C: {C}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = SVC(kernel=kernel, C=C, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    train_acc = clf.score(X_train_scaled, y_train)
    print(f"  Training accuracy: {train_acc:.2%}")
    
    return clf, scaler

def evaluate_model(clf, scaler, X_test, y_test, time_ranges, save_dir):
    """
    Evaluate model performance and generate visualizations.
    
    Args:
        clf: trained classifier
        scaler: standardizer
        X_test: test features
        y_test: test labels
        time_ranges: list of time ranges
        save_dir: output directory
    """
    print(f"\nEvaluate model performance:")
    
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = clf.predict(X_test_scaled)
    
    test_acc = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {test_acc:.2%}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nClassification report:")
    class_names = [f"Class {i+1}\n[{start}-{end}]" 
                   for i, (start, end) in enumerate(time_ranges)]
    report = classification_report(y_test, y_pred, 
                                   target_names=class_names,
                                   digits=3)
    print(report)
    
    plot_confusion_matrix(cm, class_names, test_acc, save_dir)
    
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    print(f"\nAccuracy per class:")
    for i, acc in enumerate(class_accuracies):
        print(f"  Class {i+1} [{time_ranges[i][0]}-{time_ranges[i][1]}]: {acc:.2%}")
    
    return test_acc, cm, y_pred

def plot_confusion_matrix(cm, class_names, accuracy, save_dir):
    """Plot confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=[f"Pred\n{name}" for name in class_names],
                yticklabels=[f"True\n{name}" for name in class_names],
                ax=ax, cbar_kws={'label': 'Percentage (%)'},
                annot_kws={'fontsize': 18, 'fontweight': 'bold'})
    
    ax.set_title(f'Confusion Matrix - SVM Time Classification\n'
                f'Overall Accuracy: {accuracy:.2%}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'svm_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Confusion matrix saved: {save_path}")

def cross_validation(X, y, n_folds=5):
    """
    Cross-validation to assess model stability.
    
    Args:
        X: all features
        y: all labels
        n_folds: number of folds
    """
    print(f"\n{n_folds}-fold cross-validation:")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=n_folds)
    
    print(f"  Fold accuracies: {[f'{s:.2%}' for s in scores]}")
    print(f"  Mean accuracy: {scores.mean():.2%} (+/- {scores.std():.2%})")
    
    return scores

def main():
    """Main function."""
    print("\n" + "="*70)
    print("SVM Time Classification Experiment - Lap Counting Data")
    print("="*70)
    
    data_path = os.path.join(load_path, "lap_counting_2_activity.npy")
    save_dir = os.path.join(plot_dir, "figs_svm_time_classification")
    
    test_size = 4500
    random_seed = 42
    
    if not os.path.exists(data_path):
        print(f"\nError: data file not found: {data_path}")
        return
    
    print(f"\nStep 1: Load data")
    activity = load_data(data_path)
    print(f"  Data shape: {activity.shape} (episodes, time, neurons)")
    
    print(f"\nStep 2: Extract time slices (use middle 5 time steps)")
    slices_list, time_ranges, actual_ranges = extract_time_slices(
        activity, use_middle_only=True, middle_size=5
    )
    
    print(f"\nStep 3: Prepare training and test data")
    X_train, X_test, y_train, y_test, train_idx, test_idx = prepare_dataset(
        slices_list, test_size=test_size, random_seed=random_seed
    )
    
    print(f"\nStep 4: Train SVM classifier")
    clf, scaler = train_svm(X_train, y_train, kernel='linear', C=1.0)
    
    print(f"\nStep 5: Evaluate model")
    test_acc, cm, y_pred = evaluate_model(
        clf, scaler, X_test, y_test, actual_ranges, save_dir
    )
    
    print(f"\nStep 6: Cross-validation (use all data)")
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    cv_scores = cross_validation(X_all, y_all, n_folds=5)
    
    print(f"\nStep 7: Save summary")
    summary_path = os.path.join(save_dir, 'classification_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SVM Time Classification Results\n")
        f.write("="*70 + "\n\n")
        f.write("Data info:\n")
        f.write(f"  Data shape: {activity.shape}\n")
        f.write(f"  Training episodes: {len(train_idx)}\n")
        f.write(f"  Test episodes: {len(test_idx)}\n")
        f.write(f"  Test indices: {test_idx}\n\n")
        
        f.write("Time slices:\n")
        for i in range(len(time_ranges)):
            orig_start, orig_end = time_ranges[i]
            actual_start, actual_end = actual_ranges[i]
            f.write(f"  Class {i+1}: original [{orig_start}, {orig_end}] -> used [{actual_start}, {actual_end}]\n")
        f.write(f"\n")
        
        f.write("SVM parameters:\n")
        f.write(f"  Kernel: linear\n")
        f.write(f"  C: 1.0\n\n")
        
        f.write("Performance metrics:\n")
        f.write(f"  Test accuracy: {test_acc:.2%}\n")
        f.write(f"  Cross-validation accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})\n\n")
        
        f.write("Confusion matrix:\n")
        f.write(f"{cm}\n\n")
        
        f.write("Accuracy per class:\n")
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracies):
            f.write(f"  Class {i+1} [{actual_ranges[i][0]}-{actual_ranges[i][1]}]: {acc:.2%}\n")
    
    print(f"  Summary saved: {summary_path}")
    
    print(f"\n{'='*70}")
    print(f"Experiment complete!")
    print(f"{'='*70}")
    print(f"\nKey results:")
    print(f"  Test accuracy: {test_acc:.2%}")
    print(f"  Cross-validation accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    print(f"\nConclusion:")
    if test_acc > 0.8:
        print("  SVM can distinguish different time slices well based on neural activity patterns.")
    elif test_acc > 0.5:
        print("  SVM can partially distinguish different time slices, but accuracy is moderate.")
    else:
        print("  SVM struggles to distinguish different time slices based on neural activity patterns.")
    
    print(f"\nAll results saved to: {save_dir}/")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

