# -*- coding: utf-8 -*-
"""
Sklearn surrogate HTTP API (Python 3.7).

Keep task_id / joblib filenames in sync with ``bbo/tasks/dbtune/catalog.py``.
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request

# 1. 解决 Random Forest 路径问题
try:
    import sklearn.ensemble.forest
    sys.modules['sklearn.ensemble._forest'] = sys.modules['sklearn.ensemble.forest']
except ImportError:
    pass

# 2. 解决 Decision Tree 路径问题 (本次的新报错)
try:
    import sklearn.tree.tree
    sys.modules['sklearn.tree._classes'] = sys.modules['sklearn.tree.tree']
except ImportError:
    pass

# --- 与 catalog.SURROGATE_BENCHMARKS 同步 ---
# 每项: (joblib, objective, maximize, env_override, knobs_json 文件名) — 解码在容器内用 knobs + X-name
TASK_DEFS = {
    "knob_surrogate_sysbench_5": (
        "RF_SYSBENCH_5knob.joblib",
        "throughput",
        True,
        "AGENTIC_BBO_SYSBENCH5_SURROGATE",
        "knobs_SYSBENCH_top5.json",
    ),
    "knob_surrogate_sysbench_all": (
        "SYSBENCH_all.joblib",
        "throughput",
        True,
        "AGENTIC_BBO_SYSBENCH_ALL_SURROGATE",
        "knobs_mysql_all_197.json",
    ),
    "knob_surrogate_job_5": (
        "RF_JOB_5knob.joblib",
        "latency",
        False,
        "AGENTIC_BBO_JOB5_SURROGATE",
        "knobs_JOB_top5.json",
    ),
    "knob_surrogate_job_all": (
        "JOB_all.joblib",
        "latency",
        False,
        "AGENTIC_BBO_JOB_ALL_SURROGATE",
        "knobs_mysql_all_197.json",
    ),
    "knob_surrogate_pg_5": (
        "pg_5.joblib",
        "throughput",
        True,
        "AGENTIC_BBO_PG5_SURROGATE",
        "knobs_pg_top5.json",
    ),
    "knob_surrogate_pg_20": (
        "pg_20.joblib",
        "throughput",
        True,
        "AGENTIC_BBO_PG20_SURROGATE",
        "knobs_pg_top20.json",
    ),
}

# 占位模型：与 catalog.resolve_bundled_joblib_path 中 sysbench_5 回退一致
_PLACEHOLDER_SYSBENCH5 = "sysbench_5knob_surrogate.joblib"

ASSETS_DIR = os.environ.get("SURROGATE_ASSETS_DIR", os.path.join(os.path.dirname(__file__), "assets"))

_eval_lock = threading.RLock()
# model, X-names, joblib 路径, KnobSpaceFromJson（[0,1] -> 物理量）
_models = {}  # type: Dict[str, Tuple[Any, List[str], str, Any]]

app = Flask(__name__)


def _resolve_joblib_path(task_id: str, default_name: str, env_var: Optional[str]) -> str:
    if env_var:
        v = os.environ.get(env_var)
        if v:
            return os.path.expanduser(v)
    primary = os.path.join(ASSETS_DIR, default_name)
    if os.path.isfile(primary):
        return primary
    if task_id == "knob_surrogate_sysbench_5":
        tiny = os.path.join(ASSETS_DIR, _PLACEHOLDER_SYSBENCH5)
        if os.path.isfile(tiny):
            return tiny
    return primary


def _load_payload(path: str):
    import joblib

    p = path
    if not os.path.isfile(p):
        raise FileNotFoundError("Surrogate .joblib not found: {0}".format(p))
    payload = joblib.load(p)
    if not isinstance(payload, dict):
        raise ValueError("Expected joblib dict with model and X-name")
    model = payload.get("model")
    names = payload.get("X-name")
    if model is None or names is None:
        raise ValueError("joblib must contain 'model' and 'X-name'")
    if isinstance(names, np.ndarray):
        names = names.tolist()
    nlist = [str(x) for x in names]
    return model, nlist


def _get_model_and_space(task_id: str):
    with _eval_lock:
        if task_id in _models:
            return _models[task_id]
        if task_id not in TASK_DEFS:
            raise KeyError("Unknown task_id: {0}".format(task_id))
        default_name, _obj, _mx, env_var, knobs_fn = TASK_DEFS[task_id]
        path = _resolve_joblib_path(task_id, default_name, env_var)
        model, names = _load_payload(path)
        kpath = os.path.join(ASSETS_DIR, knobs_fn)
        if not os.path.isfile(kpath):
            raise FileNotFoundError("Missing knobs json: {0}".format(kpath))
        from knob_space import KnobSpaceFromJson

        kspace = KnobSpaceFromJson(kpath, list(names))
        if kspace.dim != len(names):
            raise ValueError("Knob space dim {0} vs names {1}".format(kspace.dim, len(names)))
        _models[task_id] = (model, names, path, kspace)
        return _models[task_id]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/task/<task_id>", methods=["GET"])
def task_metadata(task_id):
    try:
        m = _get_model_and_space(task_id)
    except Exception as ex:
        # 503: 服务在但资源未就绪（缺 .joblib / knobs json 等），避免与“路由 404”混淆
        return jsonify({"status": "error", "message": str(ex)}), 503
    _model, names, path, _kspace = m
    _file, objective, maximize, _envk, _kfn = TASK_DEFS[task_id]
    return jsonify(
        {
            "status": "ok",
            "task_id": task_id,
            "feature_names": names,
            "objective_name": objective,
            "maximize": maximize,
            "joblib_path": path,
            "input_contract": "POST /evaluate with json body task_id and x (d floats in [0,1] each); y is the objective.",
        }
    )


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """主路径: ``{"task_id": "...", "x": [0..1]^d}`` -> ``y``。可选 ``features``(物理) 与旧版兼容。"""
    payload = request.get_json(silent=True) or {}
    task_id = payload.get("task_id")
    if not task_id or not isinstance(task_id, (str, bytes)):
        return jsonify({"status": "error", "message": "Missing 'task_id'"}), 400
    if isinstance(task_id, bytes):
        task_id = task_id.decode("utf-8")

    with _eval_lock:
        try:
            m = _get_model_and_space(task_id)
        except Exception as ex:
            return jsonify({"status": "error", "message": str(ex)}), 400
        model, names, _path, kspace = m
        _fn, obj_name, _maximize, _e, _kjson = TASK_DEFS[task_id]

        # (A) 归一化 x in [0,1]^d — 与 BBO 搜索空间直接对应，推荐
        if "x" in payload and payload.get("x") is not None:
            xn = payload.get("x")
            if not isinstance(xn, list):
                return jsonify({"status": "error", "message": "'x' must be a list"}), 400
            try:
                row_norm = [float(t) for t in xn]
            except (TypeError, ValueError) as ex:
                return jsonify({"status": "error", "message": "Invalid x: {0!s}".format(ex)}), 400
            if len(row_norm) != len(names):
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "x length {0} != d={1}".format(len(row_norm), len(names)),
                        }
                    ),
                    400,
                )
            xnorm = np.asarray(row_norm, dtype=np.float64)
            phys = kspace.decode(xnorm)
            xpred = phys.reshape(1, -1)
        else:
            # (B) 物理量 features — 与旧版 / 与 debug 工具兼容
            feats = payload.get("features")
            if not isinstance(feats, list):
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Provide 'x' (unit hypercube) or 'features' (physical).",
                        }
                    ),
                    400,
                )
            try:
                row = [float(x) for x in feats]
            except (TypeError, ValueError) as ex:
                return jsonify({"status": "error", "message": "Invalid features: {0!s}".format(ex)}), 400
            if len(row) != len(names):
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Feature dim mismatch: expect {0}, got {1}".format(
                                len(names), len(row)
                            ),
                        }
                    ),
                    400,
                )
            xpred = np.asarray(row, dtype=np.float64).reshape(1, -1)

        y = model.predict(xpred)
        yf = float(np.asarray(y).ravel()[0])

    return jsonify(
        {
            "status": "success",
            "y": yf,
            obj_name: yf,
            "objective": obj_name,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8090"))
    app.run(host="0.0.0.0", port=port, threaded=True)
