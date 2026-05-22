#!/usr/bin/env python3
"""
detect_env.py — deterministic ML environment probe.

Emits a machine-readable JSON "environment manifest" describing the host's
accelerator, toolkit, interconnect, and package managers, plus a list of
risk FLAGS that downstream reasoning (the SKILL) turns into can/can't verdicts.

Design rules:
  * Pure Python stdlib only. Runs on Python 3.8+. No third-party imports.
  * Never crash. Every probe is wrapped; missing tools => null/"unknown",
    not an exception. A CPU-only laptop and an 8xH100 node both produce
    valid JSON.
  * FACTS only. The script does NOT decide whether flash-attn will build;
    it reports compute capability, arch, CUDA, python, OS — and raises FLAGS
    for known-risky combinations. The verdict layer lives in the SKILL +
    references so the compatibility knowledge stays easy to update.

Usage:
    python3 detect_env.py                 # pretty JSON to stdout
    python3 detect_env.py --out env_report.json
    python3 detect_env.py --quiet         # JSON only, no stderr notes
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _run(cmd: list[str], timeout: int = 8) -> Optional[str]:
    """Run a command, return stdout text or None. Never raises."""
    try:
        if shutil.which(cmd[0]) is None:
            return None
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if out.returncode != 0 and not out.stdout.strip():
            return None
        return out.stdout
    except Exception:
        return None


def _has(tool: str) -> bool:
    return shutil.which(tool) is not None


def _ver(tool: str, *args: str) -> Optional[str]:
    out = _run([tool, *args])
    if not out:
        return None
    return out.strip().splitlines()[0] if out.strip() else None


# --------------------------------------------------------------------------- #
# compute-capability -> NVIDIA arch codename + minimum CUDA toolkit
# This is the one place we hardcode a small, slow-moving table. The detailed
# torch/wheel matrix lives in references/cuda-torch-matrix.md.
# --------------------------------------------------------------------------- #

_NV_ARCH = [
    # (min_cc_inclusive, max_cc_exclusive, codename, min_cuda)
    (5.0, 6.0, "maxwell", "9.0"),
    (6.0, 7.0, "pascal", "9.0"),
    (7.0, 7.2, "volta", "9.0"),
    (7.2, 7.6, "turing", "10.0"),
    (8.0, 8.6, "ampere", "11.0"),
    (8.6, 8.9, "ampere", "11.1"),
    (8.9, 9.0, "ada-lovelace", "11.8"),
    (9.0, 10.0, "hopper", "11.8"),
    (10.0, 12.0, "blackwell", "12.8"),   # GB100/B100/B200 sm_100
    (12.0, 13.0, "blackwell", "12.8"),   # GB10/RTX 50 / sm_120
]


def _nv_arch(cc: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """compute capability '12.1' -> ('blackwell', '12.8')."""
    if not cc:
        return None, None
    try:
        v = float(cc)
    except ValueError:
        return None, None
    for lo, hi, name, mincuda in _NV_ARCH:
        if lo <= v < hi:
            return name, mincuda
    if v >= 13.0:
        return "post-blackwell", "13.0"
    return "pre-maxwell", None


# --------------------------------------------------------------------------- #
# accelerator probes — one per vendor, each returns dict or None
# --------------------------------------------------------------------------- #


def probe_nvidia() -> Optional[dict[str, Any]]:
    if not _has("nvidia-smi"):
        return None
    q = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if not q:
        return None
    gpus = []
    for line in q.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, mem, driver, cc = parts[0], parts[1], parts[2], parts[3]
        try:
            vram = round(float(mem) / 1024, 1) if mem.replace(".", "").isdigit() else None
        except ValueError:
            vram = None
        gpus.append({"name": name, "vram_gb": vram, "compute_cap": cc})
    if not gpus:
        return None
    driver = gpus and _run([
        "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
    ])
    driver = driver.strip().splitlines()[0].strip() if driver else None
    cc = gpus[0]["compute_cap"]
    codename, min_cuda = _nv_arch(cc)
    homogeneous = len({g["name"] for g in gpus}) == 1
    return {
        "vendor": "nvidia",
        "name": gpus[0]["name"],
        "count": len(gpus),
        "homogeneous": homogeneous,
        "vram_gb": gpus[0]["vram_gb"],
        "compute_cap": cc,
        "arch_codename": codename,
        "min_cuda_for_arch": min_cuda,
        "gpus": gpus,
    }


def probe_amd_rocm() -> Optional[dict[str, Any]]:
    # Covers AMD Instinct/Radeon and Hygon DCU (ROCm-compatible stack).
    if not (_has("rocm-smi") or _has("rocminfo")):
        return None
    name = None
    info = _run(["rocminfo"]) or ""
    m = re.search(r"Marketing Name:\s*(.+)", info)
    if m:
        name = m.group(1).strip()
    count = info.count("Device Type:                             GPU") or None
    return {
        "vendor": "amd-or-hygon-rocm",
        "name": name,
        "count": count,
        "note": "ROCm stack. If this is a Hygon DCU, see references/chips/hygon-dcu.md.",
    }


def probe_apple() -> Optional[dict[str, Any]]:
    if platform.system() != "Darwin":
        return None
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or ""
    is_arm = platform.machine() == "arm64"
    return {
        "vendor": "apple",
        "name": chip.strip() or ("Apple Silicon" if is_arm else "Intel Mac"),
        "count": 1 if is_arm else 0,
        "backend": "mps" if is_arm else "cpu",
        "note": "Apple Silicon -> torch MPS backend. No CUDA. See references/chips/apple-mps.md.",
    }


def probe_ascend() -> Optional[dict[str, Any]]:
    # Huawei Ascend (昇腾) — npu-smi is the canonical tool.
    if not _has("npu-smi"):
        return None
    out = _run(["npu-smi", "info"]) or ""
    names = re.findall(r"\b(\d+)\s+(\d+)\s+\d+\s+([0-9]+[A-Za-z].*?)\s", out)
    count = len(re.findall(r"\b910[A-Za-z0-9]*\b|\b310[A-Za-z0-9]*\b", out)) or None
    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    return {
        "vendor": "huawei-ascend",
        "name": "Ascend NPU",
        "count": count,
        "cann_home": cann,
        "note": "昇腾 NPU -> CANN + torch_npu. See references/chips/ascend.md.",
    }


def probe_cambricon() -> Optional[dict[str, Any]]:
    # Cambricon (寒武纪) MLU — cnmon is the canonical tool.
    if not _has("cnmon"):
        return None
    out = _run(["cnmon"]) or ""
    count = len(re.findall(r"MLU\d+", out)) or None
    return {
        "vendor": "cambricon",
        "name": "Cambricon MLU",
        "count": count,
        "note": "寒武纪 MLU -> Cambricon PyTorch (torch_mlu). See references/chips/cambricon.md.",
    }


def probe_metax() -> Optional[dict[str, Any]]:
    # MetaX (沐曦) — mx-smi.
    if not _has("mx-smi"):
        return None
    out = _run(["mx-smi"]) or ""
    return {
        "vendor": "metax",
        "name": "MetaX GPU",
        "count": out.count("GPU") or None,
        "note": "沐曦 GPU -> MACA stack. See references/chips/metax.md.",
    }


def probe_tpu() -> Optional[dict[str, Any]]:
    # Cloud TPU — no smi tool; detect by the TPU device node / libtpu / env.
    # NOTE: /dev/accel0 is the TPU node. Do NOT key off /dev/vfio — that is a
    # generic IOMMU device present on many non-TPU Linux hosts (false positive).
    signals = []
    if os.path.exists("/dev/accel0"):
        signals.append("device-node")
    try:
        if any("libtpu" in f for f in os.listdir("/lib") if isinstance(f, str)):
            signals.append("libtpu")
    except Exception:
        pass
    if os.environ.get("TPU_NAME") or os.environ.get("COLAB_TPU_ADDR"):
        signals.append("env")
    if not signals:
        return None
    return {
        "vendor": "google-tpu",
        "name": "Cloud TPU",
        "count": None,
        "signals": signals,
        "note": "TPU -> JAX (preferred) or torch_xla. See references/chips/tpu.md.",
    }


def detect_accelerator() -> dict[str, Any]:
    for probe in (
        probe_nvidia,
        probe_ascend,
        probe_cambricon,
        probe_metax,
        probe_amd_rocm,
        probe_apple,
        probe_tpu,
    ):
        try:
            res = probe()
        except Exception:
            res = None
        if res:
            return res
    return {"vendor": "cpu", "name": "CPU only", "count": 0,
            "note": "No accelerator detected. CPU-only workloads. See references/chips/apple-mps.md isn't relevant; use CPU torch."}


# --------------------------------------------------------------------------- #
# toolkit, interconnect, package managers, installed frameworks
# --------------------------------------------------------------------------- #


def detect_toolkit(accel: dict[str, Any]) -> dict[str, Any]:
    cuda_nvcc = None
    out = _run(["nvcc", "--version"])
    if out:
        m = re.search(r"release\s+([\d.]+)", out)
        cuda_nvcc = m.group(1) if m else None

    # nvidia-smi reports the *driver's* max supported CUDA, which can differ
    # from the installed toolkit (nvcc). Both matter; report both.
    cuda_driver_max = None
    smi = _run(["nvidia-smi"])
    if smi:
        m = re.search(r"CUDA Version:\s*([\d.]+)", smi)
        cuda_driver_max = m.group(1) if m else None

    driver = None
    d = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if d:
        driver = d.strip().splitlines()[0].strip()

    # cuDNN: best-effort header scan.
    cudnn = None
    for p in ("/usr/include/cudnn_version.h",
              "/usr/local/cuda/include/cudnn_version.h"):
        try:
            with open(p) as f:
                txt = f.read()
            major = re.search(r"#define CUDNN_MAJOR (\d+)", txt)
            minor = re.search(r"#define CUDNN_MINOR (\d+)", txt)
            patch = re.search(r"#define CUDNN_PATCHLEVEL (\d+)", txt)
            if major:
                cudnn = ".".join(g.group(1) for g in (major, minor, patch) if g)
                break
        except Exception:
            continue

    rocm = _ver("hipcc", "--version") if _has("hipcc") else None
    return {
        "cuda_toolkit_nvcc": cuda_nvcc,
        "cuda_driver_max": cuda_driver_max,
        "driver_version": driver,
        "cudnn": cudnn,
        "rocm": rocm,
    }


def detect_interconnect(accel: dict[str, Any]) -> dict[str, Any]:
    count = accel.get("count") or 0   # count may be None for some accelerators
    res: dict[str, Any] = {
        "nvlink": False,
        "nvlink_detail": None,
        "topology": "single-gpu" if count <= 1 else "multi-gpu",
        "multi_node": None,
        "multi_node_signals": [],
    }
    if accel.get("vendor") == "nvidia" and count > 1:
        topo = _run(["nvidia-smi", "topo", "-m"])
        if topo:
            has_nv = bool(re.search(r"\bNV\d+\b", topo))
            res["nvlink"] = has_nv
            res["nvlink_detail"] = "NVLink present (NV# in topo)" if has_nv \
                else "PCIe/SYS only — no NVLink between GPUs"
            res["topology"] = "multi-gpu-nvlink" if has_nv else "multi-gpu-pcie"

    # multi-node hints from common launchers/schedulers.
    sig = []
    for var in ("WORLD_SIZE", "SLURM_JOB_NUM_NODES", "SLURM_NNODES",
                "PBS_NUM_NODES", "OMPI_COMM_WORLD_SIZE", "NNODES"):
        if os.environ.get(var):
            sig.append(f"{var}={os.environ[var]}")
    if _has("sinfo") or _has("scontrol"):
        sig.append("slurm-present")
    res["multi_node_signals"] = sig
    if sig:
        res["multi_node"] = "likely (scheduler/env detected — confirm)"
    return res


def detect_pkg_managers() -> dict[str, Any]:
    mgrs = {}
    for tool in ("uv", "conda", "mamba", "micromamba", "pip", "pixi", "poetry"):
        if _has(tool):
            mgrs[tool] = _ver(tool, "--version")
    return {
        "available": list(mgrs.keys()),
        "versions": mgrs,
        "in_conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "in_virtualenv": os.environ.get("VIRTUAL_ENV"),
    }


def detect_frameworks() -> dict[str, Any]:
    """Probe the *current* python for already-installed ML frameworks."""
    res: dict[str, Any] = {}
    code = (
        "import json,sys; d={}\n"
        "try:\n import torch; d['torch']={'version':torch.__version__,"
        "'cuda':getattr(torch.version,'cuda',None),"
        "'cuda_available':torch.cuda.is_available(),"
        "'device_count':(torch.cuda.device_count() if torch.cuda.is_available() else 0),"
        "'mps':bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())}\n"
        "except Exception as e: d['torch']=None\n"
        "for m in ('transformers','tokenizers','numpy','flash_attn','xformers','vllm','deepspeed','accelerate','jax','torch_npu'):\n"
        " try:\n  mod=__import__(m); d[m]=getattr(mod,'__version__','installed')\n"
        " except Exception: pass\n"
        "print(json.dumps(d))"
    )
    out = _run([sys.executable, "-c", code], timeout=25)
    if out:
        try:
            res = json.loads(out.strip().splitlines()[-1])
        except Exception:
            res = {"_parse_error": True}
    return res


# --------------------------------------------------------------------------- #
# risk FLAGS — deterministic signals; the SKILL turns these into verdicts
# --------------------------------------------------------------------------- #


def compute_flags(report: dict[str, Any]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []

    def add(level: str, code: str, msg: str) -> None:
        flags.append({"level": level, "code": code, "message": msg})

    plat = report["platform"]
    accel = report["accelerator"]
    tk = report["toolkit"]

    arch = plat.get("arch", "")
    if arch in ("aarch64", "arm64") and accel.get("vendor") == "nvidia":
        add("warn", "arm-cuda",
            "aarch64 + NVIDIA: many prebuilt wheels (flash-attn, xformers, "
            "bitsandbytes) ship x86_64 only. Expect source builds or unavailability.")

    pyv = plat.get("python", "")
    try:
        pymajmin = tuple(int(x) for x in pyv.split(".")[:2])
    except Exception:
        pymajmin = (0, 0)
    if pymajmin >= (3, 13):
        add("warn", "python-too-new",
            f"Python {pyv}: newest ML wheels often lag. flash-attn/vllm/some "
            "torch builds may have no cp313 wheel. 3.10–3.12 is the safe band.")
    if pymajmin and pymajmin < (3, 9):
        add("warn", "python-too-old",
            f"Python {pyv}: below 3.9, modern torch/transformers drop support.")

    if accel.get("vendor") == "nvidia":
        codename = accel.get("arch_codename")
        min_cuda = accel.get("min_cuda_for_arch")
        cuda = tk.get("cuda_toolkit_nvcc") or tk.get("cuda_driver_max")
        if codename == "blackwell":
            add("warn", "blackwell",
                f"Blackwell (sm_{(accel.get('compute_cap') or '').replace('.','')}): "
                "needs CUDA >= 12.8 and a recent torch (>=2.7 / nightly cu128). "
                "FlashAttention-2 lacks sm_120 kernels — use PyTorch SDPA or FA3-class paths.")
        if min_cuda and cuda:
            try:
                if float(cuda) < float(min_cuda):
                    add("error", "cuda-too-old",
                        f"CUDA {cuda} < {min_cuda} required for {codename}. "
                        "GPU kernels won't run. Upgrade toolkit/driver.")
            except ValueError:
                pass
        if not cuda:
            add("warn", "no-cuda-toolkit",
                "No nvcc found. Prebuilt torch wheels bundle their own CUDA "
                "runtime, but any source build (flash-attn/apex) needs a "
                "matching CUDA toolkit installed.")

    fw = report.get("frameworks", {})
    torch = fw.get("torch")
    if torch and torch.get("cuda") and accel.get("vendor") == "nvidia":
        if not torch.get("cuda_available"):
            add("error", "torch-no-cuda",
                "torch is installed but torch.cuda.is_available() is False — "
                "CPU-only build or driver/toolkit mismatch. Reinstall the GPU build.")
    if torch and (transf := fw.get("transformers")) and (tok := fw.get("tokenizers")):
        add("info", "transformers-tokenizers",
            f"transformers {transf} pins a tokenizers range; installed {tok}. "
            "Upgrading one without the other is the #1 silent break — see env-isolation.md.")
    if fw.get("numpy", "").startswith("2."):
        add("info", "numpy2",
            "numpy 2.x present: many ML packages compiled against numpy 1.x "
            "ABI break at import. Pin numpy<2 if you hit '_ARRAY_API not found'.")

    if not flags:
        add("info", "ok", "No high-risk environment signals detected.")
    return flags


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def build_report() -> dict[str, Any]:
    accel = detect_accelerator()
    report: dict[str, Any] = {
        "schema": "ml-env-probe/1",
        "platform": {
            "os": platform.system().lower(),
            "arch": platform.machine(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "kernel": platform.release(),
            "cpu_count": os.cpu_count(),
        },
        "accelerator": accel,
        "toolkit": detect_toolkit(accel),
        "interconnect": detect_interconnect(accel),
        "pkg_managers": detect_pkg_managers(),
        "frameworks": detect_frameworks(),
    }
    # memory (Linux best-effort)
    try:
        with open("/proc/meminfo") as f:
            m = re.search(r"MemTotal:\s+(\d+)", f.read())
            if m:
                report["platform"]["ram_gb"] = round(int(m.group(1)) / 1024 / 1024, 1)
    except Exception:
        pass
    report["flags"] = compute_flags(report)
    return report


def main() -> int:
    out_path = None
    quiet = "--quiet" in sys.argv
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        if i + 1 < len(sys.argv):
            out_path = sys.argv[i + 1]

    report = build_report()
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if out_path:
        try:
            with open(out_path, "w") as f:
                f.write(text + "\n")
            if not quiet:
                print(f"\n# wrote {out_path}", file=sys.stderr)
        except Exception as e:
            print(f"# failed to write {out_path}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
