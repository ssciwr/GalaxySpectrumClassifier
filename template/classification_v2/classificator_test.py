import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt


class LineRatioAGNClassifier:
    """
    Train HII vs AGN classifier from emission-line ratio tables.

    Input:
        df_hii : pandas.DataFrame
        df_agn : pandas.DataFrame

    Rows = objects / grid points
    Columns = named line-ratio features (e.g. 'log_OIII_Hb')
    """

    def __init__(
        self,
        model_type="rf",
        test_size=0.25,
        random_state=42,
        calibrate=True,
        calibration_method="isotonic",
        verbose=True,
        rf_params=None,
        gb_params=None,
    ):
        self.model_type = model_type.lower()
        self.test_size = test_size
        self.random_state = random_state
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.verbose = verbose

        self.rf_params = rf_params or {}
        self.gb_params = gb_params or {}

        self.pipeline_ = None
        self.is_fitted_ = False
        self.feature_names_ = None

    # --------------------------------------------------
    # Model builder
    # --------------------------------------------------
    def _build_model(self):
        preproc = SimpleImputer(strategy="median")

        if self.model_type in ["rf", "random_forest"]:
            base_model = RandomForestClassifier(
                n_estimators=self.rf_params.get("n_estimators", 600),
                max_depth=self.rf_params.get("max_depth", None),
                min_samples_leaf=self.rf_params.get("min_samples_leaf", 1),
                max_features=self.rf_params.get("max_features", "sqrt"),
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.random_state,
            )

        elif self.model_type in ["gb", "hgb", "gradient_boosting"]:
            base_model = HistGradientBoostingClassifier(
                learning_rate=self.gb_params.get("learning_rate", 0.05),
                max_depth=self.gb_params.get("max_depth", 5),
                max_iter=self.gb_params.get("max_iter", 500),
                min_samples_leaf=self.gb_params.get("min_samples_leaf", 20),
                random_state=self.random_state,
            )

        else:
            raise ValueError("model_type must be 'rf' or 'gb'.")

        pipe = Pipeline(
            steps=[
                ("imputer", preproc),
                ("clf", base_model),
            ]
        )

        if self.calibrate:
            pipe = CalibratedClassifierCV(
                estimator=pipe,
                method=self.calibration_method,
                cv=3,
            )

        return pipe

    # --------------------------------------------------
    # Input handling
    # --------------------------------------------------
    @staticmethod
    def _validate_input_tables(df_hii, df_agn):
        if not isinstance(df_hii, pd.DataFrame) or not isinstance(df_agn, pd.DataFrame):
            raise TypeError("Inputs must be pandas DataFrames.")

        if not df_hii.columns.equals(df_agn.columns):
            raise ValueError("HII and AGN tables must have identical feature columns.")

        return df_hii.copy(), df_agn.copy()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    def fit(self, df_hii, df_agn, evaluate=True):
        df_hii, df_agn = self._validate_input_tables(df_hii, df_agn)

        self.feature_names_ = list(df_hii.columns)

        X = np.vstack([df_hii.values, df_agn.values])
        y = np.hstack(
            [
                np.zeros(len(df_hii), dtype=int),
                np.ones(len(df_agn), dtype=int),
            ]
        )

        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.pipeline_ = self._build_model()
        self.pipeline_.fit(self.X_train_, self.y_train_)
        self.is_fitted_ = True

        if evaluate and self.verbose:
            self._evaluate(self.X_test_, self.y_test_)

        return self

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    def _evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        print("\n=== Model evaluation (hold-out test set) ===")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=["HII", "AGN"]))
        print(f"ROC AUC = {roc_auc_score(y_test, y_proba):.4f}")

    # --------------------------------------------------
    # Prediction interface
    # --------------------------------------------------
    def predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted.")

        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names_].values

        return self.pipeline_.predict_proba(X)

    def predict(self, X, threshold=0.5):
        p_agn = self.predict_proba(X)[:, 1]
        return (p_agn >= threshold).astype(int)

    def classify(self, X, threshold_agn=0.8, threshold_hii=0.2):
        p_agn = self.predict_proba(X)[:, 1]
        labels = np.full(len(p_agn), "Composite/Uncertain", dtype=object)
        labels[p_agn >= threshold_agn] = "AGN"
        labels[p_agn <= threshold_hii] = "HII"
        return labels, p_agn

    # --------------------------------------------------
    # ROC curve
    # --------------------------------------------------
    def plot_roc_curve(self, ax=None, label=None):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted.")

        p_agn = self.predict_proba(self.X_test_)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test_, p_agn)
        auc = roc_auc_score(self.y_test_, p_agn)

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))

        lbl = label or f"ROC (AUC={auc:.3f})"
        ax.plot(fpr, tpr, lw=2, label=lbl)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)

        return auc

    def plot_permutation_importance(self, n_repeats=30, scoring="roc_auc", ax=None):
        """
        Plot permutation feature importance using ROC AUC.

        Importance = decrease in ROC AUC when a feature is shuffled.

        Parameters
        ----------
        n_repeats : int
            Number of shuffles per feature (>=20 recommended)
        scoring : str
            Scoring metric ('roc_auc' recommended)
        ax : matplotlib axis, optional

        Returns
        -------
        importances : pandas.Series
            Mean importance per feature (indexed by feature name)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted.")

        if not hasattr(self, "X_test_"):
            raise RuntimeError("No test set available. Fit the model first.")

        result = permutation_importance(
            self.pipeline_,
            self.X_test_,
            self.y_test_,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=-1,
        )

        importances = pd.Series(
            result.importances_mean,
            index=self.feature_names_,
            name="Δ ROC AUC",
        )

        std = result.importances_std

        # Sort for plotting
        importances = importances.sort_values()

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))

        ax.barh(
            importances.index,
            importances.values,
            xerr=std[np.argsort(importances.values)],
            color="tab:green",
            alpha=0.8,
        )

        ax.set_xlabel("Decrease in ROC AUC")
        ax.set_title("Permutation feature importance")
        ax.grid(alpha=0.3)

        return importances


def plot_bpt_nii(
    NII_Ha, OIII_Hb, ax=None, labels=None, colors=None, alpha=0.6, s=20, show_lines=True
):
    """
    Plot [NII]-BPT diagram.

    Parameters
    ----------
    OIII_Hb : array-like
        log10([OIII]5007 / Hβ)
    NII_Ha : array-like
        log10([NII]6584 / Hα)
    ax : matplotlib axis (optional)
    labels : str or array-like (optional)
        Labels for legend
    colors : color or array-like (optional)
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        NII_Ha, OIII_Hb, s=s, c=colors, alpha=alpha, edgecolor="none", label=labels
    )

    if show_lines:
        x = np.linspace(-2.0, 0.6, 500)

        # Kauffmann (2003)
        y_kauff = 0.61 / (x - 0.05) + 1.3
        ax.plot(x, y_kauff, "k--", lw=1.5, label="Kauffmann 2003")

        # Kewley (2001)
        y_kew = 0.61 / (x - 0.47) + 1.19
        ax.plot(x, y_kew, "k-", lw=1.5, label="Kewley 2001")

    ax.set_xlabel(r"log([NII]$\lambda6584$/H$\alpha$)")
    ax.set_ylabel(r"log([OIII]$\lambda5007$/H$\beta$)")

    ax.set_xlim(-2.0, 0.6)
    ax.set_ylim(-1.5, 1.5)

    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # if ax is None:
    plt.show()

    return ax


def prepare_full_line_features_hbeta_norm(
    df_raw, line_list=["OIII_4363", "OIII_4959", "OIII_5007", "Ha"], fluxHb=100.0
):
    """
    Build feature table using all specified emission lines.

    Assumes fluxes are already normalized to Hbeta.
    Features = log10(line_flux / Hbeta) = log10(line_flux).
    """

    df = df_raw.copy()

    features = pd.DataFrame(index=df.index)

    for line in line_list:
        if line in df.columns:
            flux = pd.to_numeric(df[line], errors="coerce")

            with np.errstate(divide="ignore", invalid="ignore"):
                flux = flux.where(flux > 0, np.nan)
                features[f"log_{line}"] = np.log10(flux / fluxHb)
        else:
            features[f"log_{line}"] = np.nan

    # Add physically useful combined lines

    # SII sum
    if "SII6717" in df.columns and "SII6731" in df.columns:
        SII_sum = pd.to_numeric(df["SII6717"], errors="coerce") + pd.to_numeric(
            df["SII6731"], errors="coerce"
        )
        SII_sum = SII_sum.where(SII_sum > 0, np.nan)
        features["log_SII_sum"] = np.log10(SII_sum / fluxHb)

    # OIII sum
    if "OIII_4959" in df.columns and "OIII_5007" in df.columns:
        OIII_sum = pd.to_numeric(df["OIII_4959"], errors="coerce") + pd.to_numeric(
            df["OIII_5007"], errors="coerce"
        )
        OIII_sum = OIII_sum.where(OIII_sum > 0, np.nan)
        features["log_OIII_sum"] = np.log10(OIII_sum / fluxHb)

    # OII 3726+3729
    if "OII_3726" in df.columns and "OII_3729" in df.columns:
        OII_sum = pd.to_numeric(df["OII_3726"], errors="coerce") + pd.to_numeric(
            df["OII_3729"], errors="coerce"
        )
        OII_sum = OII_sum.where(OII_sum > 0, np.nan)
        features["log_OII3726_29_sum"] = np.log10(OII_sum / fluxHb)

    return features


def read_cloudy_grid(path):
    """
    Read a Cloudy 17.00 model grid (.dat) as written for this project.

    Each file has a block of '#'-prefixed comment lines, then a whitespace-
    separated column header and numeric rows. The first three columns
    (12+log(O/H), log(N/O), log(U)) are model parameters; the remaining columns
    are emission-line fluxes already normalised to Hbeta = 1. Zero fluxes mean
    the line is absent in that model and become NaN in the feature builder.
    """
    return pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        comment="#",
        na_values=["nan", "NaN"],
    )


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Emission-line columns present in the Cloudy grid files. The first three
    # columns (12+log(O/H), log(N/O), log(U)) are model parameters and are
    # intentionally excluded so only line fluxes become features.
    line_list = [
        "OII_3727",
        "NeIII_3868",
        "OIII_4363",
        "HeI_4471",
        #        "HeII_4686",
        "OIII_5007",
        "NII_5755",
        "HeI_5876",
        "SIII_6312",
        "NII_6584",
        "HeI_6678",
        "SII_6717,31",
        "ArIII_7135",
        "OII_7325",
        "SIII_9069",
    ]
    basepath = "../../data"
    hii_paths = sorted(glob.glob(f"{basepath}/C17_POPSTAR*.dat"))
    agn_paths = sorted(glob.glob(f"{basepath}/C17_AGN*.dat"))

    if not hii_paths or not agn_paths:
        raise FileNotFoundError(
            "Expected Cloudy grids in the working directory: "
            "C17_POPSTAR*.dat (HII region models) and C17_AGN*.dat (AGN models)."
        )

    print(f"Reading {len(hii_paths)} HII grid file(s): {hii_paths}")
    df_hii_raw = pd.concat(
        [read_cloudy_grid(path) for path in hii_paths], ignore_index=True
    )

    print(f"Reading {len(agn_paths)} AGN grid file(s): {agn_paths}")
    df_agn_raw = pd.concat(
        [read_cloudy_grid(path) for path in agn_paths], ignore_index=True
    )

    # build features (grid fluxes are already normalised to Hbeta = 1)
    df_hii = prepare_full_line_features_hbeta_norm(
        df_hii_raw, line_list=line_list, fluxHb=1.0
    )
    df_agn = prepare_full_line_features_hbeta_norm(
        df_agn_raw, line_list=line_list, fluxHb=1.0
    )

    print(df_hii)

    # Align feature columns between the two sets (union, fill missing with NaN)
    all_cols = sorted(set(df_hii.columns).union(df_agn.columns))
    df_hii = df_hii.reindex(columns=all_cols)
    df_agn = df_agn.reindex(columns=all_cols)

    # Optionally drop features that are mostly missing across both sets
    drop_nan_threshold = 0.5  # drop feature if >50% NaN across combined rows
    nan_frac = pd.concat([df_hii, df_agn]).isna().mean()
    drop_cols = nan_frac[nan_frac > drop_nan_threshold].index.tolist()
    if drop_cols:
        print(
            f"Dropping {len(drop_cols)} features with >{drop_nan_threshold * 100:.0f}% NaNs:",
            drop_cols,
        )
        df_hii = df_hii.drop(columns=drop_cols)
        df_agn = df_agn.drop(columns=drop_cols)
        all_cols = [c for c in all_cols if c not in drop_cols]

    print("Final feature columns used:", all_cols)

    # Now run your classifiers (LineRatioAGNClassifier must be defined/imported)
    clf_gb = LineRatioAGNClassifier(model_type="gb")
    clf_gb.fit(df_hii, df_agn)

    clf_rf = LineRatioAGNClassifier(model_type="rf")
    clf_rf.fit(df_hii, df_agn)

    # ROC plot
    fig, ax = plt.subplots(figsize=(6, 6))
    auc_rf = clf_rf.plot_roc_curve(ax=ax, label="Random Forest")
    auc_gb = clf_gb.plot_roc_curve(ax=ax, label="Gradient Boosting")
    plt.title("HII vs AGN ROC curve")
    plt.show()

    print("RF AUC =", auc_rf)
    print("GB AUC =", auc_gb)

    # Permutation importance (Gradient Boosting)
    print("\nPermutation importance (Gradient Boosting):")
    importances_gb = clf_gb.plot_permutation_importance(n_repeats=30)
    plt.show()
    print(importances_gb)

    # Classify a small new sample: ensure it has same columns
    rng = np.random.default_rng(1)
    df_new = pd.DataFrame(rng.normal(size=(10, len(all_cols))), columns=all_cols)
    labels, p_agn = clf_gb.classify(df_new, threshold_agn=0.8, threshold_hii=0.2)

    print("\nNew sample classification:")
    for i, (lab, p) in enumerate(zip(labels, p_agn)):
        print(f"Obj {i:02d}: {lab:>18s}   P(AGN)={p:.3f}")
