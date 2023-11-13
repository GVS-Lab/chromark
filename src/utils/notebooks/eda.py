import os
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    confusion_matrix,
    plot_roc_curve,
    auc,
    balanced_accuracy_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm.notebook import tqdm


def read_in_protein_dataset(
        data_dir,
        feature_file_path,
        qc_result_file_path,
        gh2ax_foci_result_file_path=None,
        gh2ax_foci_result_image_index_col="image_index",
        filter_samples: List[str] = None,
):
    all_features = []
    subdirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for subdir in tqdm(subdirs, desc="Load data"):
        feature_path = subdir + feature_file_path
        features = pd.read_csv(feature_path, index_col=0)
        features["sample"] = os.path.split(subdir)[1].split("_")[0].lower()
        features["timepoint"] = os.path.split(subdir)[1].split("_")[1]
        qc_result_path = subdir + qc_result_file_path
        qc_results = pd.read_csv(qc_result_path, index_col=0)
        features.loc[
            set(list(features.index)).intersection(qc_results.index), "qc_pass"
        ] = qc_results.loc[qc_results.index, "qc_pass"]
        features["data_dir"] = subdir
        if gh2ax_foci_result_file_path is not None:
            gh2ax_foci_path = subdir + gh2ax_foci_result_file_path
            gh2ax_results = pd.read_csv(gh2ax_foci_path, index_col=0)
            grouped_gh2ax_results = gh2ax_results.groupby(
                gh2ax_foci_result_image_index_col
            )
            count_gh2ax_results = grouped_gh2ax_results.count()
            sum_gh2ax_results = grouped_gh2ax_results.sum()
            avg_gh2ax_results = grouped_gh2ax_results.mean()
            features["gh2ax_foci_count"] = 0
            features["gh2ax_sum_foci_area"] = 0
            features["gh2ax_avg_foci_area"] = 0
            features.loc[
                set(list(count_gh2ax_results.index)).intersection(list(features.index)),
                "gh2ax_foci_count",
            ] = np.array(
                count_gh2ax_results.loc[
                    set(list(count_gh2ax_results.index)).intersection(
                        list(features.index)
                    ),
                    "label",
                ]
            )
            features.loc[
                set(list(sum_gh2ax_results.index)).intersection(list(features.index)),
                "gh2ax_sum_foci_area",
            ] = np.array(
                sum_gh2ax_results.loc[
                    set(list(count_gh2ax_results.index)).intersection(
                        list(features.index)
                    ),
                    "area",
                ]
            )
            features.loc[
                set(list(avg_gh2ax_results.index)).intersection(list(features.index)),
                "gh2ax_avg_foci_area",
            ] = np.array(
                avg_gh2ax_results.loc[
                    set(list(count_gh2ax_results.index)).intersection(
                        list(features.index)
                    ),
                    "area",
                ]
            )

        if (
                filter_samples is None
                or np.unique(features.loc[:, "sample"])[0] in filter_samples
        ):
            all_features.append(features)
    all_features_df = all_features[0].copy()
    for i in range(1, len(all_features)):
        all_features_df = all_features_df.append(all_features[i])
    return all_features_df


def read_in_marker_dataset(
        data_dir,
        feature_file_path,
        qc_result_file_path,
        marker_label_file_path,
        filter_samples: List[str] = None,
        gh2ax_foci_result_file_path=None,
        gh2ax_foci_result_image_index_col="image_index",
):
    all_features = []
    subdirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for subdir in tqdm(subdirs, desc="Load data"):
        feature_path = subdir + feature_file_path
        features = pd.read_csv(feature_path, index_col=0)
        features["sample"] = os.path.split(subdir)[1].split("_")[0].lower()
        features["timepoint"] = os.path.split(subdir)[1].split("_")[1]
        qc_result_path = subdir + qc_result_file_path

        qc_results = pd.read_csv(qc_result_path, index_col=0)
        features.loc[
            set(list(features.index)).intersection(qc_results.index), "qc_pass"
        ] = qc_results.loc[qc_results.index, "qc_pass"]
        features["data_dir"] = subdir

        if gh2ax_foci_result_file_path is not None:
            gh2ax_foci_path = subdir + gh2ax_foci_result_file_path
            gh2ax_results = pd.read_csv(gh2ax_foci_path, index_col=0)
            grouped_gh2ax_results = gh2ax_results.groupby(
                gh2ax_foci_result_image_index_col
            )
            count_gh2ax_results = grouped_gh2ax_results.count()
            sum_gh2ax_results = grouped_gh2ax_results.sum()
            avg_gh2ax_results = grouped_gh2ax_results.mean()
            features["gh2ax_foci_count"] = 0
            features["gh2ax_sum_foci_area"] = 0
            features["gh2ax_avg_foci_area"] = 0
            features.loc[
                set(list(count_gh2ax_results.index)).intersection(list(features.index)),
                "gh2ax_foci_count",
            ] = np.array(
                count_gh2ax_results.loc[
                    set(list(count_gh2ax_results.index)).intersection(
                        list(features.index)
                    ),
                    "label",
                ]
            )
            features.loc[
                set(list(sum_gh2ax_results.index)).intersection(list(features.index)),
                "gh2ax_sum_foci_area",
            ] = np.array(
                sum_gh2ax_results.loc[
                    set(list(count_gh2ax_results.index)).intersection(
                        list(features.index)
                    ),
                    "area",
                ]
            )
            features.loc[
                set(list(avg_gh2ax_results.index)).intersection(list(features.index)),
                "gh2ax_avg_foci_area",
            ] = np.array(
                avg_gh2ax_results.loc[
                    set(list(count_gh2ax_results.index)).intersection(
                        list(features.index)
                    ),
                    "area",
                ]
            )

        marker_labels = pd.read_csv(subdir + marker_label_file_path, index_col=0, )
        features = features.merge(marker_labels, left_index=True, right_index=True)
        if (
                filter_samples is None
                or np.unique(features.loc[:, "sample"])[0] in filter_samples
        ):
            all_features.append(features)
    all_features_df = all_features[0].copy()
    for i in range(1, len(all_features)):
        all_features_df = all_features_df.append(all_features[i])
    return all_features_df


def read_in_data(
        feature_file_path: str, qc_file_path: str, sample: str, timepoint: str = "tp0"
):
    features = pd.read_csv(feature_file_path, index_col=0)
    qc_results = pd.read_csv(qc_file_path, index_col=0)
    features.loc[qc_results.index, "qc_pass"] = qc_results.loc[
        qc_results.index, "qc_pass"
    ]
    features["sample"] = sample
    features["timepoint"] = timepoint
    return features


def preprocess_data(
        data,
        drop_columns=[
            "label",
            "centroid-0",
            "centroid-1",
            "orientation",
            "weighted_centroid-0",
            "weighted_centroid-1",
        ],
        remove_constant_features=False,
):
    filtered_data = data.loc[data["qc_pass"] == True]
    print(
        "Nuclei that did not pass the quality check: {}/{}. Remaining: {}.".format(
            len(data) - len(filtered_data), len(data), len(filtered_data),
        )
    )
    if remove_constant_features:
        data = filtered_data.loc[:, (filtered_data != filtered_data.iloc[0]).any()]
    else:
        data = filtered_data
    data = data.dropna(axis=1)
    print(
        "Removed {} constant or features with missing values. Remaining: {}.".format(
            len(filtered_data.columns) - len(data.columns), len(data.columns)
        )
    )
    cleaned_data = data.drop(
        columns=list(set(drop_columns).intersection(set(data.columns)))
    )
    print(
        "Removed additional {} features. Remaining: {}.".format(
            len(data.columns) - len(cleaned_data.columns), len(cleaned_data.columns)
        )
    )
    if not cleaned_data.index.is_unique:
        new_idc = []
        idx_count = {}
        idc = list(cleaned_data.index)
        for i in range(len(cleaned_data)):
            idx = idc[i]
            if idx not in idx_count:
                new_idc.append(idx)
                idx_count[idx] = 1
            else:
                new_idc.append("{}_{}".format(idx, idx_count[idx]))
                idx_count[idx] = idx_count[idx] + 1
        cleaned_data.index = np.array(new_idc).astype(str)
    return cleaned_data


def compute_vif(data):
    vifs = pd.DataFrame()
    vifs["feature"] = data.columns
    vifs["vif"] = [
        variance_inflation_factor(data.values, i) for i in range(data.shape[1])
    ]
    return vifs.sort_values("vif", ascending=False)


def remove_collinear_features(data, threshold=5):
    to_drop = []
    # Sequentially drop features with VIF > threshold
    vifs = compute_vif(data)
    print(vifs.head(5))
    while np.max(vifs.loc[:, "vif"]) > threshold:
        to_drop.append(vifs.loc[vifs["vif"] == np.max(vifs.loc[:, "vif"]), "feature"])
        vifs = compute_vif(data.drop(columns=[to_drop[-1]]))

    print(
        "Removed {}/{} features with a VIF above {}. Remaining: {}".format(
            len(to_drop),
            len(data.columns),
            threshold,
            len(data.columns) - len(to_drop),
        )
    )
    return data.drop(to_drop, axis=1)


def remove_correlated_features(data, threshold):
    data_corr = data.corr().abs()
    upper = data_corr.where(np.triu(np.ones(data_corr.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(
        "Removed {}/{} features with a Pearson correlation above {}. Remaining: {}"
        .format(
            len(to_drop),
            len(data.columns),
            threshold,
            len(data.columns) - len(to_drop),
        )
    )
    return data.drop(to_drop, axis=1)


def plot_feature_importance(
        importance,
        names,
        model_type,
        figsize=[6, 4],
        cmap=["gray"],
        n_features=10,
        feature_color_dict=None,
        labelsize=6,
):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)
    fi_df = fi_df.head(n_features)
    # Define size of bar plot
    fig, ax = plt.subplots(figsize=figsize)
    # Plot Searborn bar chart
    ax = sns.barplot(
        y=fi_df["feature_importance"], x=fi_df["feature_names"], palette=cmap, ax=ax
    )
    if feature_color_dict is not None:
        for xticklabel in ax.get_xticklabels():
            xticklabel.set_color(feature_color_dict[xticklabel.get_text()])
            xticklabel.set_rotation(90)
    ax.tick_params(axis="x", labelsize=labelsize)
    ax.tick_params(axis="y", labelsize=labelsize)

    # xticklabel.set_ha("right")
    # Add chart labels
    ax.set_title(model_type + "FEATURE IMPORTANCE")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig, ax


def plot_roc_for_stratified_cv(
        X, y, n_splits, classifier, title, pos_label=None, groups=None
):
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits)
    else:
        cv = StratifiedGroupKFold(n_splits=n_splits)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(8, 8))
    if groups is None:
        for i, (train, test) in enumerate(cv.split(X, y)):
            classifier.fit(X[train], y[train])
            viz = plot_roc_curve(
                classifier,
                X[test],
                y[test],
                name="ROC fold {}".format(i),
                alpha=0.3,
                lw=1,
                ax=ax,
                pos_label=pos_label,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
    else:
        for i, (train, test) in enumerate(cv.split(X, y, groups=groups)):
            classifier.fit(X[train], y[train])
            viz = plot_roc_curve(
                classifier,
                X[test],
                y[test],
                name="ROC fold {}".format(i),
                alpha=0.3,
                lw=1,
                ax=ax,
                pos_label=pos_label,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.legend(loc="lower right")
    return fig, ax, classifier


def compute_cv_conf_mtx(
        model,
        n_folds,
        features,
        labels,
        groups=None,
        balance_train=False,
        random_state=1234,
):
    features = np.array(features)
    labels = np.array(labels)
    n_classes = len(np.unique(labels))

    confusion_mtx = np.zeros([n_classes, n_classes])
    rus = RandomUnderSampler(random_state=random_state)

    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=n_folds)
        for train_index, test_index in skf.split(features, labels, groups):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            if balance_train:
                X_train, y_train = rus.fit_resample(X_train, y_train)

            model.fit(X_train, y_train)
            print(Counter(y_test))
            fold_confusion_matrix = confusion_matrix(
                y_test, model.predict(X_test), normalize=None, labels=model.classes_,
            )
            confusion_mtx += fold_confusion_matrix
    else:
        skf = StratifiedKFold(n_splits=n_folds)
        for train_index, test_index in skf.split(features, labels):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            model.fit(X_train, y_train)
            fold_confusion_matrix = confusion_matrix(
                y_test, model.predict(X_test), normalize=None, labels=model.classes_,
            )
            confusion_mtx += fold_confusion_matrix
    return pd.DataFrame(confusion_mtx, index=model.classes_, columns=model.classes_)


def get_cv_results_by_fold(
        model,
        features,
        labels,
        n_folds,
        groups=None,
        balance_train=False,
        scoring=balanced_accuracy_score,
        random_state=1234,
):
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_folds)
    else:
        cv = StratifiedKFold(n_splits=n_folds)

    rus = RandomUnderSampler(random_state=random_state)

    cv_scores = []
    cv_labels = []
    cv_preds = []
    cv_groups = []
    cv_pred_probs = []
    features = np.array(features)
    labels = np.array(labels)
    groups = np.array(groups)
    for train_idx, test_idx in cv.split(X=features, y=labels, groups=groups):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        if groups is not None and balance_train:
            X_train, y_train = rus.fit_resample(X_train, y_train)
        print(Counter(y_train))

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        pred_probs = model.predict_proba(X_test)
        cv_labels.append(y_test)
        cv_preds.append(preds)
        cv_pred_probs.append(pred_probs)
        cv_scores.append(scoring(y_test, preds))
        if groups is not None:
            cv_groups.append(np.unique(groups[test_idx]))
    return cv_scores, cv_labels, cv_preds, cv_pred_probs, cv_groups


def summarize_group_cv_results_by_fold(
        model,
        features,
        labels,
        groups,
        n_folds=None,
        balance_train=False,
        scoring=accuracy_score,
        random_state=1234,
):
    if n_folds is None:
        n_folds = len(np.unique(groups))
    cv = StratifiedGroupKFold(n_splits=n_folds)
    rus = RandomUnderSampler(random_state=random_state)

    result = {
        "group": [],
        "score": [],
        "avg_max_pred_prob": [],
        "avg_true_class_pred_prob": [],
        "majority_class": [],
        "majority_predicted_class": [],
    }

    features = np.array(features)
    labels = np.array(labels)
    groups = np.array(groups)
    for train_idx, test_idx in cv.split(X=features, y=labels, groups=groups):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        if balance_train:
            X_train, y_train = rus.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)
        classes = model.classes_
        preds = model.predict(X_test)
        pred_probs = model.predict_proba(X_test)
        test_groups = np.unique(groups[test_idx])
        test_group = "_".join(sorted(list(test_groups)))
        score = scoring(y_test, preds)
        for c in classes:
            if "prop_"+str(c) in result:
                result["prop_"+str(c)].append(np.mean(preds == c))
            else:
                result["prop_"+str(c)] = [np.mean(preds==c)]
        avg_max_pred_prob = np.mean(np.max(pred_probs, axis=1))
        true_class_pred_probs = []
        for i in range(len(y_test)):
            true_class_pred_probs.append(pred_probs[i, classes == y_test[i]])
        avg_true_class_pred_prob = np.mean(true_class_pred_probs)

        result["group"].append(test_group)
        result["score"].append(score)
        result["avg_max_pred_prob"].append(avg_max_pred_prob)
        result["avg_true_class_pred_prob"].append(avg_true_class_pred_prob)
        result["majority_class"].append(Counter(y_test).most_common(1)[0][0])
        result["majority_predicted_class"].append(Counter(preds).most_common(1)[0][0])

    return pd.DataFrame(result, index=list(range(n_folds)))


def compute_pairwise_emd(features, labels):
    unique_labels = np.unique(labels)
    n_unique_labels = len(unique_labels)

    emd_mtx = np.zeros((n_unique_labels, n_unique_labels))
    for i in tqdm(range(n_unique_labels)):
        label_i_features = features[labels == unique_labels[i]]
        a = np.ones((len(label_i_features),))
        a = a / len(a)
        for j in range(n_unique_labels):
            if i == j:
                emd_mtx[i, j] = 0
            else:
                label_j_features = features[labels == unique_labels[j]]
                b = np.ones((len(label_j_features),))
                b = b / len(b)
                M = ot.dist(label_i_features, label_j_features)
                emd_mtx[i, j] = ot.emd2(a, b, M)
    emd_mtx = pd.DataFrame(emd_mtx, index=unique_labels, columns=unique_labels)
    return emd_mtx


def get_permute_group_labels(labels, groups):
    unique_groups = np.unique(groups)
    unique_labels = []
    for group in unique_groups:
        unique_labels.append(np.unique(labels[groups == group])[0])
    unique_group_label_dict = dict(zip(unique_groups, unique_labels))
    perm_unique_group_label_dict = dict(
        zip(unique_groups, np.random.permutation(unique_labels))
    )
    perm_labels = []
    for group in groups:
        perm_labels.append(perm_unique_group_label_dict[group])
    return perm_labels, perm_unique_group_label_dict, unique_group_label_dict


def run_nuclei_ablation_study_cv(
    estimator,
    features,
    labels,
    groups,
    n_repeats=5,
    balance_train=True,
    scale_features=True,
    n_folds=10,
    random_state=1234,
):
    np.random.seed(random_state)

    rus = RandomUnderSampler(random_state=random_state)

    if scale_features:
        sc = StandardScaler()
        features = pd.DataFrame(
            sc.fit_transform(features), index=features.index, columns=features.columns
        )

    features = np.array(features)
    labels = np.array(labels)
    groups = np.array(groups)

    results = {"n_nuclei": [], "frac_nuclei": [], "sample": [], "lopo_accuracy": []}
    for i in tqdm(np.arange(0.1, 1.1, 0.1), position=0):
        for j in tqdm(range(n_repeats), position=1):
            skf = StratifiedGroupKFold(n_splits=n_folds)
            accs = []
            for train_index, test_index in skf.split(features, labels, groups):
                X_train, X_test = features[train_index], features[test_index]
                y_train, y_test = labels[train_index], labels[test_index]

                train_groups = groups[train_index]

                n_nuclei = min(list(dict(Counter(train_groups)).values()))
                unique_train_groups = list(np.unique(train_groups))
                n_sample_dict = dict(
                    zip(
                        unique_train_groups,
                        [int(n_nuclei * i)] * len(unique_train_groups),
                    )
                )
                #                 print(Counter(y_train))
                if i < 1:
                    rus2 = RandomUnderSampler(sampling_strategy=n_sample_dict)

                    sample_idc = np.arange(0, len(X_train)).reshape((-1, 1))
                    sample_idc, sample_train_groups = rus2.fit_resample(
                        sample_idc, train_groups
                    )

                    X_train = X_train[sample_idc.ravel()]
                    y_train = y_train[sample_idc.ravel()]

                #                 print(Counter(y_train))

                if balance_train:
                    X_train, y_train = rus.fit_resample(X_train, y_train)

                estimator.fit(X_train, y_train)
                acc = accuracy_score(y_test, estimator.predict(X_test))
                accs.append(acc)

            results["n_nuclei"].append(n_nuclei)
            results["frac_nuclei"].append(i)
            results["sample"].append(j)
            results["lopo_accuracy"].append(np.mean(accs))

    results = pd.DataFrame(results)
    return results


def run_patient_ablation_study_cv(estimator, features, labels, groups, n_repeats=5, balance_train=True,
                                  scale_features=True, n_folds=10, random_state=1234):
    np.random.seed(random_state)
    classes = np.unique(labels)

    rus = RandomUnderSampler(random_state=random_state)

    if scale_features:
        sc = StandardScaler()
        features = pd.DataFrame(
            sc.fit_transform(features), index=features.index, columns=features.columns
        )

    features = np.array(features)
    labels = np.array(labels)
    groups = np.array(groups)

    patient_label_dict = {}
    for c in classes:
        patient_label_dict[c] = np.unique(groups[labels == c])

    results = {"n_train_patients": [], "sample": [], "lopo_accuracy": []}
    for i in tqdm(range(1, len(patient_label_dict[classes[0]])), position=0):
        for j in tqdm(range(n_repeats), position=1):
            skf = StratifiedGroupKFold(n_splits=n_folds)
            accs = []
            for train_index, test_index in skf.split(features, labels, groups):
                X_train, X_test = features[train_index], features[test_index]
                y_train, y_test = labels[train_index], labels[test_index]


                train_groups = groups[train_index]
                selected_train_patients = []
                for c in classes:
                    class_train_patients = np.unique(train_groups[y_train == c])
                    selected_train_patients.extend(
                            list(np.random.choice(class_train_patients, size=i, replace=False)))

                train_mask = []
                for train_group in train_groups:
                    train_mask.append(train_group in selected_train_patients)
                X_train = X_train[train_mask]
                y_train = y_train[train_mask]

                if balance_train:
                    X_train, y_train = rus.fit_resample(X_train, y_train)

                estimator.fit(X_train, y_train)
                acc = accuracy_score(y_test, estimator.predict(X_test))
                accs.append(acc)

            #                 if j == 0:
            #                     print(Counter(train_groups[train_mask]))
            #                     print(Counter(y_test))

            results["n_train_patients"].append(i)
            results["sample"].append(j)
            results["lopo_accuracy"].append(np.mean(accs))

    results = pd.DataFrame(results)
    return results
