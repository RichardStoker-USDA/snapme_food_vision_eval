# Serving benchmark models on Kubernetes

The harness sends image-and-text requests to a model endpoint. A model served on Kubernetes can be evaluated
through the same interface as a model in a Slurm job, provided the endpoint accepts the registered request fields.

`tools/k8s_manifest.py` creates a Deployment and Service from a registry entry. It reads the project label
from `project.toml` and writes the manifest as JSON. Review the output for the destination cluster before
applying it.

```bash
pixi run python tools/k8s_manifest.py qwen3-6-35b-a3b-fp8 --namespace models --gpu-product NVIDIA-L40S > model.json
kubectl apply -f model.json          # after review, through the cluster's own change process
pixi run snapme-eval run --system direct --model qwen3-6-35b-a3b-fp8 \
  --base-url http://qwen3-6-35b-a3b-fp8.models.svc:8000/v1 --out runs/snapme_v1/direct_v2/qwen3-6-35b-a3b-fp8
```

The generated manifest uses these settings:

| Item | Assumption |
|---|---|
| Weights | a PersistentVolumeClaim (`hf-cache`) holding the Hugging Face cache with the pinned snapshot; the pod runs offline |
| GPUs | `nvidia.com/gpu` from the NVIDIA device plugin; the count comes from the model's L40S or A100 plan |
| Health | `/health` for startup and readiness; startup probe allows 120 attempts at 10-second intervals |
| Shared memory | 16 GiB in-memory `emptyDir` at `/dev/shm` for tensor-parallel workers |

The generator targets the NVIDIA device-plugin resource interface. Clusters using another allocation
mechanism need a site-specific manifest; the generator does not configure that mechanism.

Models with `extra_pip` or `python_shim` in their file need those added to the pod (an init container that
installs the packages into a shared volume, and the shim folder mounted on `PYTHONPATH`); the tool writes a warning to standard error when these fields are present. The tool writes a manifest and deploys nothing.
