"""
evaluate_model.py
-----------------
Final check on the untouched 20 % test set, plus a rough look at which
features the model relies on.

    python evaluate_model.py
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, recall_score)

from preprocess import prepare_data, CLASSES, FEATURES, MODELS, REPORTS, SEED

FALL_ID = CLASSES.index("FALL")


def permutation_importance(model, X, y, n_repeats=10):
    """Drop in accuracy / FALL recall when one feature is shuffled."""
    rng = np.random.default_rng(SEED)
    base_pred = model.predict(X, verbose=0).argmax(1)
    base_acc = accuracy_score(y, base_pred)
    base_fall = recall_score(y, base_pred, labels=[FALL_ID], average=None)[0]
    rows = []
    for j, name in enumerate(FEATURES):
        acc_drops, fall_drops = [], []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            p = model.predict(Xp, verbose=0).argmax(1)
            acc_drops.append(base_acc - accuracy_score(y, p))
            fall_drops.append(base_fall - recall_score(y, p, labels=[FALL_ID], average=None)[0])
        rows.append((name, np.mean(acc_drops), np.mean(fall_drops)))
    return sorted(rows, key=lambda r: -r[1])


def main():
    REPORTS.mkdir(exist_ok=True)
    data = prepare_data(verbose=False)
    X_test, y_test = data["X_test"], data["y_test"]

    model = tf.keras.models.load_model(MODELS / "helmet_mlp.keras")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = y_prob.argmax(1)

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    acc = accuracy_score(y_test, y_pred)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    out(f"Test samples : {len(y_test)}")
    out(f"Accuracy     : {acc:.4f}")
    out(f"Weighted     : precision {p_w:.4f}  recall {r_w:.4f}  F1 {f_w:.4f}")
    out(f"Macro        : precision {p_m:.4f}  recall {r_m:.4f}  F1 {f_m:.4f}")
    out()
    out("Classification report")
    out(classification_report(y_test, y_pred, target_names=CLASSES, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    out("Confusion matrix (rows = true, columns = predicted)")
    out("order: " + ", ".join(CLASSES))
    out(str(cm))
    out()

    # --- FALL focus ---
    fall_total = int((y_test == FALL_ID).sum())
    fall_missed = int(((y_test == FALL_ID) & (y_pred != FALL_ID)).sum())
    fall_false_alarm = int(((y_test != FALL_ID) & (y_pred == FALL_ID)).sum())
    out("FALL class")
    out(f"  real falls in test set        : {fall_total}")
    out(f"  missed (false negatives)      : {fall_missed}  ({fall_missed / fall_total:.1%})")
    out(f"  false alarms (false positives): {fall_false_alarm}")
    # a missed FALL that was still flagged as some other alert still raises an alarm
    fall_rows = y_test == FALL_ID
    fall_flag_any = int((fall_rows & (y_pred != 0)).sum())
    out(f"  falls where ANY alert fired   : {fall_flag_any}/{fall_total}")
    out()
    out("Why false negatives matter more than false positives for FALL:")
    out("  - A false negative means a miner is hurt (maybe unconscious, trapped) and")
    out("    nobody is told. The cost is delayed rescue, which can mean death.")
    out("  - A false positive means a spurious alarm. The cost is a supervisor")
    out("    checking on a miner who is fine - annoying, but harmless.")
    out("  - So for deployment, bias the decision toward catching falls, e.g. raise an")
    out("    alert when P(FALL) > ~0.3 rather than waiting until FALL is the arg-max,")
    out("    and accept more false alarms.")
    out()

    # --- what gets confused ---
    out("Most common confusions (true -> predicted):")
    pairs = [(cm[i, j], CLASSES[i], CLASSES[j]) for i in range(len(CLASSES))
             for j in range(len(CLASSES)) if i != j and cm[i, j] > 0]
    for n, t, p in sorted(pairs, reverse=True)[:6]:
        out(f"  {t:18s} -> {p:18s}: {n}")
    out()

    # --- threshold look for FALL ---
    out("Effect of lowering the FALL alert threshold (P(FALL) >= t, otherwise argmax):")
    for t in (0.5, 0.3, 0.2, 0.1):
        flag = (y_prob[:, FALL_ID] >= t) | (y_pred == FALL_ID)
        rec = (flag & fall_rows).sum() / fall_total
        fp = int((flag & ~fall_rows).sum())
        out(f"  t = {t:.1f}: FALL recall {rec:.3f}, false alarms {fp}")
    out()

    # --- feature importance ---
    out("Permutation importance on the test set (bigger = model relies on it more)")
    out(f"  {'feature':12s} {'acc drop':>9s} {'FALL recall drop':>17s}")
    imp = permutation_importance(model, X_test, y_test)
    for name, d_acc, d_fall in imp:
        out(f"  {name:12s} {d_acc:9.4f} {d_fall:17.4f}")

    (REPORTS / "evaluation_report.txt").write_text("\n".join(lines))

    plt.figure(figsize=(6.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (test set)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=130)
    plt.close()

    names = [r[0] for r in imp][::-1]
    vals = [r[1] for r in imp][::-1]
    plt.figure(figsize=(6, 4))
    plt.barh(names, vals)
    plt.xlabel("Accuracy drop when shuffled")
    plt.title("Permutation feature importance")
    plt.tight_layout()
    plt.savefig(REPORTS / "feature_importance.png", dpi=130)
    plt.close()
    print("\nSaved evaluation_report.txt, confusion_matrix.png, feature_importance.png in reports/")


if __name__ == "__main__":
    main()
