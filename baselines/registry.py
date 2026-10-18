"""Stable public method IDs; no fixed experimental seeds."""

METHODS = {
    "descriptor_bo_mean": {"engine": "descriptor_bo", "label": "Descriptor BO (Mean)", "acquisition": "Mean"},
    "descriptor_bo_pi": {"engine": "descriptor_bo", "label": "Descriptor BO (PI)", "acquisition": "PI"},
    "descriptor_bo_ei": {"engine": "descriptor_bo", "label": "Descriptor BO (EI)", "acquisition": "EI"},
    "descriptor_bo_lcb": {"engine": "descriptor_bo", "label": "Descriptor BO (LCB)", "acquisition": "LCB"},
    "descriptor_bo_ucb": {"engine": "descriptor_bo", "label": "Descriptor BO (UCB)", "acquisition": "UCB"},
    "random_forest": {"engine": "random_forest", "label": "Random Forest"},
    "tpe_24": {"engine": "tpe", "label": "Tree-structured Parzen Estimator (TPE-24)"},
    "random_search": {"engine": "random_search", "label": "Random Search"},
    "descriptor_transformer": {"engine": "descriptor_transformer", "label": "Descriptor Transformer"},
    "tabpfn": {"engine": "tabpfn", "label": "Tabular Prior-data Fitted Network (TabPFN)"}
}
ALL_METHODS = ["spectral_state_navigation", *METHODS]
