import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, accuracy_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from acquisition import get_data_stream
from processing import EEGProcessor

# --- Label mapping ---
# Physionet uses old-style, e.g. "Sleep stage 1", and new-style "Sleep stage N1" labels.
# Stage 3 and 4 are combined into N3 following AASM 2007 guidelines.
LABEL_DICT = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "R",
    # Some datasets use the new-style labels directly:
    "Sleep stage N1": "N1",
    "Sleep stage N2": "N2",
    "Sleep stage N3": "N3",
}

# The "?" label is used for unknown/unscored epochs, and we will ignore these during training.

VALID_LABELS = ["W", "N1", "N2", "N3", "R"]
EPOCH_LENGTH_SEC = 30 # standard PSG epoch length

def extract_epochs(raw, annotations, proc):
    """Extracts 30-second epochs and their corresponding expert annotations 
    from the raw data. Extracts band-power features for each epoch, and returns
    features and labels.

    Every epoch has a 6-element feature vector:
    delta_power - relative power in 0.5-3.9 Hz band => (N3 marker)
    theta_power - relative power in 4-7.9 Hz band => (N1/drowsiness marker)
    alpha_power - relative power in 8-11.9 Hz band => (N2 marker, waked/relaxed)
    beta_power  - relative power in 12-30 Hz band => (Wake/active marker)
    delta_rel   - delta_power / (alpha_power + beta_power) => (Sleep depth ratio)
    log_var     - log of signal variance, as a general marker of activity level => (Amplitude scale-invariant) 
    """
    fs = int(raw.info['sfreq'])
    epoch_samples = EPOCH_LENGTH_SEC * fs
    features, labels = [], []
    
    for annot in annotations:
        stage_label = LABEL_DICT.get(annot['description'], "?")
        if stage_label not in VALID_LABELS:
            continue  # skip unknown/unscored epochs
        
        onset_sample = int(annot['onset'] * fs)
        end_sample = onset_sample + epoch_samples
        if end_sample > len(raw.times):
            continue # skip incomplete epochs at the end
        
        epoch_data, _ = raw[0, onset_sample:end_sample]
        epoch_flat = epoch_data.flatten()
        filtered = proc.apply_filter(epoch_flat)

        # Skip artifacted epochs based on simple amplitude thresholding (can be improved with more sophisticated methods)
        if proc.check_for_artifacts(filtered):
            continue

        feats = _compute_features(filtered, fs, proc)
        features.append(feats)
        labels.append(stage_label)

    return np.array(features), np.array(labels)

def _compute_features(signal, fs, proc):
    """Computes the 6 features vector for a given EEG filtered epoch.
    Identical to the update module's _extract_features(), but here we also pass 
    the proc object to apply the same filter before feature extraction.
    This ensures that the features used for training match the features computed
    in real-time during the simulation.
    <Could be refactored to avoid code duplication, but keeping it simple for now.>
    """
    from scipy.signal import welch
    
    freqs, psd = welch(signal, fs, nperseg=fs*2)

    def bandpower(low, high):
        mask = (freqs >= low) & (freqs <= high)
        return np.trapezoid(psd[mask], freqs[mask])
    
    # Define rel band powers
    total = bandpower(0.5, 30.0)
    delta = bandpower(0.5, 3.9) / total
    theta = bandpower(4.0, 7.9) / total
    alpha = bandpower(8.0, 11.9) / total
    beta = bandpower(12.0, 30.0) / total

    denom = alpha + beta
    delta_rel = delta / denom if denom > 0 else 0
    log_var = np.log(np.var(signal) + 1e-12)

    return np.array([delta_rel, alpha, beta, log_var, theta, total])

def train_classifier(n_subjects=5, verbose=True):
    """Loads data from n_subjects, extracts epochs, trains a logistic regression classifier
    with cross-validation, prints metrics and returns the fitted pipeline.
    For cross-validation, we use StratifiedKFold with 5 folds to maintain class balance across folds.
    This is important because N1 epochs are much rarer than N2 or Wake epochs, and we want to ensure 
    the model learns to recognize them.

    Inputs:
    - n_subjects: how many subjects to load and train on (max 20 in Physionet)
    - verbose: whether to print training metrics and confusion matrix

    Outputs:
    - pipeline: sklearn.pipeline.Pipeline object containing the StandardScaler and LogisticRegression, 
    fitted on all data. Ready for predict() calls.
    - labels: the corresponding labels for the training data, useful for interpreting confusion matrices.
    """
    # Check if requested not to print verbose output
    if verbose:
        print(f"=== MLM: Loading and processing data from {n_subjects} subjects... ===")
    
    all_features, all_labels = [], []
    for subj_id in range(n_subjects):
        if verbose:
            print(f"=== MLM: Processing subject {subj_id}...", end=" ")
        raw, annotations = get_data_stream(subj_id)
        fs = int(raw.info['sfreq'])
        proc = EEGProcessor(fs)
        features, labels = extract_epochs(raw, annotations, proc)
        all_features.append(features)
        all_labels.append(labels)
        if verbose:
            print(f"Done. {len(features)} epochs extracted.")

    X = np.vstack(all_features)
    y = np.hstack(all_labels)

    if verbose:
        print(f"\n Total epochs: {len(y)}")
        unique, counts = np.unique(y, return_counts=True)
        for stage, count in zip(unique, counts):
            print(f"  {stage}: {count} epochs ({100*count/len(y):.1f}%)")
    
    # --- Build pipeline ---
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            # multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000, 
            class_weight='balanced', # corrects for N1 underrepresentation
            random_state=42
            ))
    ])

    # --- Cross-validation ---
    if verbose:
        print("\n=== MLM: Performing 5-fold cross-validation... ===")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = np.empty_like(y)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        pipeline.fit(X[train_idx], y[train_idx])
        cv_preds[val_idx] = pipeline.predict(X[val_idx])
        if verbose:
            acc = accuracy_score(y[val_idx], cv_preds[val_idx])
            print(f"  Fold {fold+1}: Accuracy = {acc:.3f}")

    # --- Final fit on all data - model returned ---
    pipeline.fit(X, y)
    label_names = list(pipeline.classes_)

    if verbose:
        print("\n=== MLM: Cross-validation results ===")
        print(classification_report(y, cv_preds, target_names=sorted(set(y))))

    # --- Plot confusion matrix ---
    _plot_results(y, cv_preds, label_names)

    return pipeline, label_names

def _plot_results(y_true, y_pred, labels):
    """Plots the confusion matrix with per-class metrics."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    ordered = [s for s in ["W", "N1", "N2", "N3", "R"] if s in labels]
    cm = confusion_matrix(y_true, y_pred, labels=ordered)

    # Per-class metrics
    precision = precision_score(y_true, y_pred, labels=ordered, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=ordered, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=ordered, average=None, zero_division=0)
    support = [np.sum(y_true == s) for s in ordered]
    accuracy = accuracy_score(y_true, y_pred)

    # --- Create figure with confusion matrix and metrics table ---
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Sleep Stage Classifier — Cross-Validation Results",
                fontsize=13, fontweight='bold'
                )
    gs = gridspec.GridSpec(2, 1,height_ratios=[3, 1], 
                           # width_ratios=[1, 1], 
                           hspace=0.5
                        )
    # --- Confusion matrix ---
    ax_cm = fig.add_subplot(gs[0])
    disp = ConfusionMatrixDisplay(cm, display_labels=ordered)
    disp.plot(ax=ax_cm, cmap='Blues', colorbar=True)
    ax_cm.set_title(f"Confusion Matrix (Overall Accuracy (5-fold CV) | Overall accuracy: {accuracy:.3f})", fontsize=11)

    # --- Metrics table ---
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')

    col_labels = ["Stage", "Precision", "Recall", "F1-Score", "Support"]
    rows = [
        [stage, f"{p:2f}", f"{r:.2f}", f"{f:.2f}", f"{s}"]
        for stage, p, r, f, s in zip(ordered, precision, recall, f1, support)
        ]
    
    # Macro averages row
    rows.append([
        "Macro Avg",
        f"{np.mean(precision):.2f}",
        f"{np.mean(recall):.2f}",
        f"{np.mean(f1):.2f}",
        f"{len(y_true)}"
    ])

    table = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Header row style
    for col_idx in range(len(col_labels)):
        table[0, col_idx].set_facecolor("#e2e7ed")
        table.scale(1, 1.5)

    # Macro average row style
    last_row = len(rows)
    for col_idx in range(len(col_labels)):
        table[last_row, col_idx].set_facecolor('#ecf0f1')
        table[last_row, col_idx].set_text_props(fontweight='bold')
 
    plt.show(block=False)


