"""
Visualize linear SVM decision boundaries and class separation on real data.

Using lap_counting_2_activity.npy, this script plots:
1. Data distribution after PCA dimensionality reduction
2. Linear SVM decision boundaries
3. Separation/distance between classes
4. Support vector distribution
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist


def load_data(path):
    """Load data and convert to real values."""
    raw = np.load(path, allow_pickle=True)
    
    if np.iscomplexobj(raw):
        data = raw.real
    else:
        data = raw
    
    data = data.astype(np.float64)
    return data


def extract_time_slices(activity, use_middle_only=True, middle_size=5):
    """Extract four time slices."""
    time_ranges = [
        (5, 24),    # Class 1
        (38, 57),   # Class 2
        (71, 90),   # Class 3
        (105, 124)  # Class 4
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
        else:
            time_slice = activity[:, start:end+1, :]
            actual_ranges.append((start, end))
        
        slices_list.append(time_slice)
    
    return slices_list, time_ranges, actual_ranges


def prepare_data_for_visualization(slices_list, test_size=4500, random_seed=42):
    """Prepare data (using the same parameters as the main script)."""
    np.random.seed(random_seed)
    
    n_episodes = slices_list[0].shape[0]
    indices = np.arange(n_episodes)
    np.random.shuffle(indices)
    
    train_indices = indices[:n_episodes - test_size]
    test_indices = indices[n_episodes - test_size:]
    
    # Build data
    X_list = []
    y_list = []
    
    for class_idx, time_slice in enumerate(slices_list):
        # Use all data (train + test) for visualization
        flat = time_slice.reshape(n_episodes, -1)
        X_list.append(flat)
        y_list.append(np.full(n_episodes, class_idx))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y, train_indices, test_indices


def plot_pca_distribution(X, y, actual_ranges, save_dir):
    """Plot data distribution after PCA reduction."""
    print("\nPlotting PCA data distribution...")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"  PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}, "
          f"total={pca.explained_variance_ratio_[:2].sum():.3f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756']
    class_names = [f'Class {i+1}\n[{start}-{end}]' 
                   for i, (start, end) in enumerate(actual_ranges)]
    
    for class_idx in range(4):
        mask = y == class_idx
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=colors[class_idx], label=class_names[class_idx],
                  alpha=0.6, s=100, edgecolors='white', linewidth=1.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                 fontsize=14, fontweight='bold')
    ax.set_title('Data Distribution in PCA Space\n(Real Neural Activity Data)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, framealpha=0.9, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'data_distribution_2d.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")
    
    return X_scaled, X_pca, pca, scaler


def plot_decision_boundary_2d(X_scaled, X_pca, y, actual_ranges, save_dir):
    """Plot linear SVM decision boundaries in 2D PCA space."""
    print("\nPlotting linear SVM decision boundaries (2D PCA space)...")
    
    # Train linear SVM in PCA space
    clf_pca = SVC(kernel='linear', C=1.0, random_state=42)
    clf_pca.fit(X_pca, y)
    
    train_acc = clf_pca.score(X_pca, y)
    print(f"  Training accuracy in PCA space: {train_acc:.2%}")
    
    # Create grid
    h = 0.02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict
    Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Decision boundary background
    colors_light = ['#B3CDE3', '#FDD0A2', '#C7E9C0', '#F6BDBD']
    cmap = ListedColormap(colors_light)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=[-0.5, 0.5, 1.5, 2.5, 3.5])
    
    # Decision boundary lines
    ax.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0.5, 1.5, 2.5], alpha=0.8)
    
    # Data points
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756']
    class_names = [f'Class {i+1} [{start}-{end}]' 
                   for i, (start, end) in enumerate(actual_ranges)]
    
    for class_idx in range(4):
        mask = y == class_idx
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=colors[class_idx], label=class_names[class_idx],
                  alpha=0.8, s=120, edgecolors='white', linewidth=2)
    
    # Support vectors
    support_vectors = X_pca[clf_pca.support_]
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
              s=200, facecolors='none', edgecolors='red', linewidths=3,
              label=f'Support Vectors (n={len(support_vectors)})')
    
    ax.set_xlabel('Principal Component 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Principal Component 2', fontsize=14, fontweight='bold')
    ax.set_title(f'Linear SVM Decision Boundaries in PCA Space\n'
                f'Training Accuracy: {train_acc:.2%}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, framealpha=0.9, loc='best')
    # ax.grid(alpha=0.3)  # Grid intentionally disabled
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'svm_decision_boundary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")
    
    return clf_pca


def plot_class_separation(X_scaled, y, actual_ranges, save_dir):
    """Analyze separation between classes."""
    print("\nAnalyzing class separation...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756']
    class_names = [f'Class {i+1}\n[{start}-{end}]' 
                   for i, (start, end) in enumerate(actual_ranges)]
    
    # 1. Distance matrix between class centers
    class_centers = []
    for class_idx in range(4):
        mask = y == class_idx
        center = X_scaled[mask].mean(axis=0)
        class_centers.append(center)
    class_centers = np.array(class_centers)
    
    # Compute Euclidean distances between centers
    center_distances = cdist(class_centers, class_centers, metric='euclidean')
    
    ax = axes[0]
    im = ax.imshow(center_distances, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f'C{i+1}' for i in range(4)], fontsize=11)
    ax.set_yticklabels([f'C{i+1}' for i in range(4)], fontsize=11)
    ax.set_title('Euclidean Distance Between Class Centers', 
                fontsize=13, fontweight='bold')
    
    # Add value annotations
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{center_distances[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance', fontsize=11)
    
    # 2. Within-class variance vs between-class separation
    ax = axes[1]
    
    within_class_vars = []
    for class_idx in range(4):
        mask = y == class_idx
        X_class = X_scaled[mask]
        center = class_centers[class_idx]
        var = np.mean(np.sum((X_class - center)**2, axis=1))
        within_class_vars.append(var)
    
    x_pos = np.arange(4)
    bars = ax.bar(x_pos, within_class_vars, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{i+1}' for i in range(4)], fontsize=11)
    ax.set_ylabel('Within-Class Variance', fontsize=12, fontweight='bold')
    ax.set_title('Within-Class Variance (Lower = Tighter)', 
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value annotations
    for bar, var in zip(bars, within_class_vars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Separability metric (between-class / within-class ratio)
    ax = axes[2]
    
    separability_scores = []
    for class_idx in range(4):
        mask = y == class_idx
        
        # Mean distance to other class centers
        other_centers = [class_centers[i] for i in range(4) if i != class_idx]
        between_dist = np.mean([np.linalg.norm(class_centers[class_idx] - other) 
                               for other in other_centers])
        
        # Within-class standard deviation
        within_std = np.sqrt(within_class_vars[class_idx])
        
        # Separability score
        separability = between_dist / (within_std + 1e-10)
        separability_scores.append(separability)
    
    bars = ax.bar(x_pos, separability_scores, color=colors, alpha=0.7, 
                 edgecolor='black', linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{i+1}' for i in range(4)], fontsize=11)
    ax.set_ylabel('Separability Score', fontsize=12, fontweight='bold')
    ax.set_title('Class Separability (Between-Class / Within-Class)\n(Higher = More Separable)', 
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value annotations
    for bar, score in zip(bars, separability_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Distance to nearest class
    ax = axes[3]
    
    nearest_distances = []
    nearest_classes = []
    for class_idx in range(4):
        # Find the nearest other class
        distances_to_others = []
        other_indices = []
        for other_idx in range(4):
            if other_idx != class_idx:
                distances_to_others.append(center_distances[class_idx, other_idx])
                other_indices.append(other_idx)
        
        min_dist_idx = np.argmin(distances_to_others)
        nearest_distances.append(distances_to_others[min_dist_idx])
        nearest_classes.append(other_indices[min_dist_idx])
    
    bars = ax.bar(x_pos, nearest_distances, color=colors, alpha=0.7, 
                 edgecolor='black', linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{i+1}' for i in range(4)], fontsize=11)
    ax.set_ylabel('Distance to Nearest Class', fontsize=12, fontweight='bold')
    ax.set_title('Distance to Nearest Neighbor Class\n(Higher = Better Separated)', 
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotations: nearest class
    for i, (bar, dist, nearest) in enumerate(zip(bars, nearest_distances, nearest_classes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{dist:.1f}\n→C{nearest+1}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Class Separation Analysis (Linear SVM on Real Data)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    save_path = os.path.join(save_dir, 'class_separation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")
    
    # Print summary statistics
    print(f"\nClass separation summary:")
    print(f"  Mean between-class distance: {np.mean(center_distances[np.triu_indices_from(center_distances, k=1)]):.2f}")
    print(f"  Mean within-class variance: {np.mean(within_class_vars):.2f}")
    print(f"  Mean separability score: {np.mean(separability_scores):.2f}")
    
    return center_distances, within_class_vars, separability_scores


def plot_high_dim_svm_performance(X_scaled, y, actual_ranges, save_dir):
    """Train linear SVM in high-dimensional space and visualize performance."""
    print("\nTraining linear SVM in high-dimensional space...")
    
    # Train linear SVM
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    clf.fit(X_scaled, y)
    
    train_acc = clf.score(X_scaled, y)
    print(f"  Training accuracy in high-dimensional space: {train_acc:.2%}")
    print(f"  Number of support vectors: {len(clf.support_)} / {len(X_scaled)}")
    
    # Get decision function values
    decision_values = clf.decision_function(X_scaled)
    
    # Plot decision function distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756']
    class_names = [f'Class {i+1} [{start}-{end}]' 
                   for i, (start, end) in enumerate(actual_ranges)]
    
    # For each class, plot the decision function distribution
    for plot_idx, target_class in enumerate(range(4)):
        ax = axes[plot_idx]
        
        for class_idx in range(4):
            mask = y == class_idx
            values = decision_values[mask, target_class]
            
            ax.hist(values, bins=30, alpha=0.6, color=colors[class_idx],
                   label=class_names[class_idx], edgecolor='black', linewidth=0.5)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                  label='Decision Boundary', alpha=0.8)
        ax.set_xlabel('Decision Function Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'Decision Function for Class {target_class+1}\n'
                    f'(Positive = Predicted as C{target_class+1})', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Linear SVM Decision Functions in High-Dimensional Space\n'
                f'Overall Accuracy: {train_acc:.2%}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    save_path = os.path.join(save_dir, 'rbf_kernel_explanation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")
    
    return clf, train_acc


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("Linear SVM visualization on real data")
    print("="*70)
    load_path = "/home/lugroup/Documents/Sen_Code/time_cell_submit/capsule-0490840-code/data/figure_5_SVM"
    plot_dir = "/home/lugroup/Documents/Sen_Code/time_cell_submit/capsule-0490840-code/figures"
    # Parameters (consistent with the main script)
    data_path = os.path.join(load_path, "lap_counting_2_activity.npy")
    save_dir = os.path.join(plot_dir, "figs_svm_explanation")
    os.makedirs(save_dir, exist_ok=True)
    
    test_size = 4500  # Consistent with the main script
    random_seed = 42
    
    # Load data
    print(f"\nLoading data: {data_path}")
    activity = load_data(data_path)
    print(f"  Data shape: {activity.shape}")
    
    # Extract time slices (use middle 5 time steps)
    print("\nExtracting time slices (middle 5 time steps)")
    slices_list, time_ranges, actual_ranges = extract_time_slices(
        activity, use_middle_only=True, middle_size=5
    )
    
    # Prepare data
    print("\nPreparing data")
    X, y, train_idx, test_idx = prepare_data_for_visualization(
        slices_list, test_size=test_size, random_seed=random_seed
    )
    print(f"  X shape: {X.shape} (samples, features)")
    print(f"  y shape: {y.shape}")
    print(f"  Feature dimension: {X.shape[1]}")
    
    # 1. Plot PCA distribution
    X_scaled, X_pca, pca, scaler = plot_pca_distribution(X, y, actual_ranges, save_dir)
    
    # 2. Plot decision boundaries (2D PCA space)
    clf_pca = plot_decision_boundary_2d(X_scaled, X_pca, y, actual_ranges, save_dir)
    
    # 3. Class separation analysis
    center_distances, within_vars, separability = plot_class_separation(
        X_scaled, y, actual_ranges, save_dir
    )
    
    # 4. High-dimensional SVM performance
    clf_high_dim, high_dim_acc = plot_high_dim_svm_performance(
        X_scaled, y, actual_ranges, save_dir
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("Visualization complete!")
    print(f"{'='*70}")
    print("\nKey findings:")
    print(f"  ✓ High-dimensional accuracy: {high_dim_acc:.2%}")
    print(f"  ✓ Mean between-class distance: {np.mean(center_distances[np.triu_indices_from(center_distances, k=1)]):.2f}")
    print(f"  ✓ Mean separability score: {np.mean(separability):.2f}")
    print(f"\nAll figures saved to: {save_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()




