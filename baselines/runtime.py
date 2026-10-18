"""Inject portable configuration before loading preserved numerical definitions."""

import importlib.util
import itertools
import json
from pathlib import Path
import sys

from spa_former.seeds import validate_seeds

COUNTER = itertools.count()
HERE = Path(__file__).resolve().parent


def engine_settings(config):
    defaults = json.loads((HERE / "configs/methods.json").read_text(encoding="utf-8"))
    seed = validate_seeds([config["seed"]])[0]
    values = {**defaults["common"], **defaults["engine_parameters"],
              "INITIAL_XLSX": Path(config["initial_file"]), "ALL_XLSX": Path(config["all_file"]),
              "OUT_DIR": Path(config["output"]), "MAX_ITERATIONS": config["max_rounds"],
              "SEED": seed, "SEEDS": [seed], "RFF_SEED": seed,
              "ACQUISITION": config.get("acquisition", "Mean"), "DEVICE": config.get("device", "auto")}
    values["VALID_SUFFIXES"] = set(values["VALID_SUFFIXES"])
    values["EPOCHS"] = config.get("epochs", 180)
    if config.get("method") == "tabpfn":
        values["N_ESTIMATORS"] = 4
    return values


def load_engine(engine, config):
    path = Path(__file__).with_name("engines") / (engine + ".py")
    name = "baselines.engines._runtime_{}".format(next(COUNTER))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.RUNTIME_CONFIG = dict(config)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(method, seed, initial_file, all_file, output, max_rounds=200, epochs=180, device="auto"):
    from .registry import METHODS

    spec = METHODS[method]
    config = {"method": method, "seed": seed, "initial_file": str(initial_file), "all_file": str(all_file),
              "output": str(output), "max_rounds": max_rounds, "epochs": epochs, "device": device,
              "acquisition": spec.get("acquisition", "Mean")}
    module = load_engine(spec["engine"], config)
    base = module.base if method == "descriptor_transformer" else module
    if method == "descriptor_transformer":
        base.property_score_from_values = module.matched_property_score_from_values
        base.property_score = module.matched_property_score
        base.predict_rows = module.matched_predict_rows
        base.train_model = module.matched_train_model
    return module, base, config
