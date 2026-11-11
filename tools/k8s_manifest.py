"""Generate a Kubernetes Deployment and Service from a registered model.

The manifest uses the registered engine image, weights revision and serving options, with a project label
from project.toml. See docs/kubernetes.md for settings that need review for each cluster.

    python tools/k8s_manifest.py qwen3-6-35b-a3b-fp8 --namespace models --gpu-product NVIDIA-L40S > model.yaml

The model cache is a PersistentVolumeClaim (default name `hf-cache`) that must already hold the pinned
snapshot (hpc/atlas/download_models.py writes the same Hugging Face cache layout). Nothing is applied.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from snapme_eval import registry
from snapme_eval.cli import serve_cmd
from snapme_eval.paths import project

CACHE = "/models/hf/hub"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--namespace", default="models")
    ap.add_argument("--pvc", default="hf-cache")
    ap.add_argument("--gpus", type=int, help="default: the model's L40S plan when --gpu-product names L40S, else A100")
    ap.add_argument("--gpu-product", default="", help="node label nvidia.com/gpu.product to pin, if any")
    a = ap.parse_args()

    m = registry.model(a.model)
    eng = registry.engines()[m["engine"]]
    plans = m.get("serve", {})
    plan = (plans.get("l40s") if "L40S" in a.gpu_product.upper() else plans.get("a100")) or {"gpus": 1}
    gpus = a.gpus or plan["gpus"]
    os.environ["HF_HUB_CACHE"] = CACHE
    cmd = serve_cmd(a.model, gpus, 8000)
    cmd = ["0.0.0.0" if x == "127.0.0.1" else x for x in cmd]   # listen on the pod address (any engine)
    for k in ("extra_pip", "python_shim"):
        if m.get(k):
            print(f"note: {a.model} has {k} = {m[k]!r}; add it to the pod (docs/kubernetes.md)", file=sys.stderr)
    name = a.model.replace(".", "-")[:50]
    image = eng["image"].removeprefix("docker://")
    selector = {"nvidia.com/gpu.product": a.gpu_product} if a.gpu_product else {}
    part_of = project()["project"]["slug"].replace("_", "-")
    labels = {"app.kubernetes.io/name": name, "app.kubernetes.io/part-of": part_of}
    deploy = {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": name, "namespace": a.namespace, "labels": labels,
                     "annotations": {"snapme_eval/hf-repo": m["hf_repo"], "snapme_eval/revision": m["revision"],
                                     "snapme_eval/license": m["license"]}},
        "spec": {"replicas": 1, "strategy": {"type": "Recreate"}, "selector": {"matchLabels": labels},
                 "template": {"metadata": {"labels": labels}, "spec": {
                     **({"nodeSelector": selector} if selector else {}),
                     "containers": [{
                         "name": "engine", "image": image, "command": cmd[:1], "args": cmd[1:],
                         "env": [{"name": "HF_HUB_OFFLINE", "value": "1"}, {"name": "HF_HUB_CACHE", "value": CACHE}],
                         "ports": [{"containerPort": 8000, "name": "http"}],
                         "resources": {"limits": {"nvidia.com/gpu": gpus}},
                         "readinessProbe": {"httpGet": {"path": "/health", "port": "http"}, "periodSeconds": 10},
                         "startupProbe": {"httpGet": {"path": "/health", "port": "http"}, "periodSeconds": 10,
                                          "failureThreshold": 120},
                         "volumeMounts": [{"name": "hf", "mountPath": "/models", "readOnly": True},
                                          {"name": "shm", "mountPath": "/dev/shm"}]}],
                     "volumes": [{"name": "hf", "persistentVolumeClaim": {"claimName": a.pvc}},
                                 {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}]}}},
    }
    svc = {"apiVersion": "v1", "kind": "Service",
           "metadata": {"name": name, "namespace": a.namespace, "labels": labels},
           "spec": {"selector": labels, "ports": [{"port": 8000, "targetPort": "http", "name": "http"}]}}
    # JSON is valid YAML, so kubectl accepts this file as is.
    print(json.dumps({"apiVersion": "v1", "kind": "List", "items": [deploy, svc]}, indent=1))
    print(f"# harness: snapme-eval run --model {a.model} --base-url http://{name}.{a.namespace}.svc:8000/v1 ...",
          file=sys.stderr)


if __name__ == "__main__":
    main()
