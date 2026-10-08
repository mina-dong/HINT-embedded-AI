"""1일차 실습용 연산량 집계와 CPU FP32 측정 도구."""

from datetime import datetime, timezone
from pathlib import Path
from numbers import Integral, Real
import hashlib
import json
import platform
import sys
import time
import uuid

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SCOPE = (
    "Conv2d + Linear only; 1 MAC = 2 ops; "
    "excludes bias, BN, activation, pooling, data movement"
)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(1024, 64), nn.ReLU(), nn.Linear(64, 8),
        )

    def forward(self, x):
        return self.net(x)


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name}은 정수여야 합니다.")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        limit = f"{minimum} 이상" if maximum is None else f"{minimum}~{maximum}"
        raise ValueError(f"{name}은 {limit}이어야 합니다.")
    return value


def load():
    """제공 가중치를 CPU FP32 평가 모드로 불러옵니다."""
    model = SmallCNN().to(device="cpu", dtype=torch.float32)
    checkpoint = torch.load(ROOT / "reference.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def inputs(size=48, batch=1):
    """32장 중 정확히 batch장(1~32)을 복사해 전처리합니다."""
    size = _integer(size, "size", 16)
    batch = _integer(batch, "batch", 1, 32)
    with np.load(ROOT / "sample_images.npz", allow_pickle=False) as data:
        if data["images"].shape[0] < batch:
            raise ValueError("제공 이미지 수가 요청한 batch보다 작습니다.")
        images = data["images"][:batch].copy()
    x = torch.from_numpy(images).permute(0, 3, 1, 2).to(dtype=torch.float32) / 255
    if tuple(x.shape[-2:]) != (size, size):
        x = nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return ((x - .5) / .5).contiguous()


def _validate_execution(model, x):
    """측정 조건을 확인하며 자동으로 장치나 자료형을 바꾸지 않습니다."""
    if not isinstance(model, nn.Module) or not isinstance(x, torch.Tensor):
        raise TypeError("model은 nn.Module, x는 torch.Tensor여야 합니다.")
    if any(module.training for module in model.modules()):
        raise ValueError("평가 모드가 필요합니다. model.eval()을 먼저 실행하세요.")
    if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
        raise ValueError("이 실습은 자동 혼합 정밀도 없이 CPU FP32로 측정합니다.")
    if x.device.type != "cpu" or x.dtype != torch.float32:
        raise ValueError("이 실습은 CPU FP32 입력만 측정합니다.")
    if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 16:
        raise ValueError("입력은 (batch, 3, height, width)이며 높이·너비는 16 이상이어야 합니다.")
    _integer(x.shape[0], "실제 batch", 1, 32)
    for value in list(model.parameters()) + list(model.buffers()):
        if value.device.type != "cpu":
            raise ValueError("이 실습은 CPU 모델만 측정합니다.")
        if value.is_complex() or (value.is_floating_point() and value.dtype != torch.float32):
            raise ValueError("모델의 부동소수점 파라미터·버퍼는 FP32여야 합니다.")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("입력에 NaN 또는 무한대가 있습니다.")


@torch.inference_mode()
def count_ops(model, x):
    """배치 전체의 Conv2d·Linear MAC/ops를 실제 출력 크기로 집계합니다."""
    _validate_execution(model, x)
    rows, handles = [], []

    def hook(name):
        def collect(layer, args, y):
            if isinstance(layer, nn.Conv2d):
                mac = y.numel() * (layer.in_channels // layer.groups)
                mac *= layer.kernel_size[0] * layer.kernel_size[1]
            elif isinstance(layer, nn.Linear):
                mac = y.numel() * layer.in_features
            else:
                return
            rows.append(dict(layer=name, type=type(layer).__name__, output=list(y.shape),
                             mac=int(mac), ops=int(2 * mac)))
        return collect

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handles.append(layer.register_forward_hook(hook(name)))
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    return rows


def excluded_layers(model):
    """집계하지 않은 말단 모듈 목록입니다. 별도 함수 연산은 탐지하지 않습니다."""
    return [dict(layer=name, type=type(layer).__name__)
            for name, layer in model.named_modules()
            if name and not list(layer.children())
            and not isinstance(layer, (nn.Conv2d, nn.Linear))]


@torch.inference_mode()
def benchmark(model, x, min_seconds=2.0, min_repeats=30, warmup=5, max_repeats=10000):
    """최소 시간·횟수를 모두 채우거나 최대 횟수까지 CPU 모델을 측정합니다."""
    _validate_execution(model, x)
    if (isinstance(min_seconds, bool) or not isinstance(min_seconds, Real)
            or not np.isfinite(min_seconds) or min_seconds <= 0):
        raise ValueError("min_seconds는 0보다 큰 유한수여야 합니다.")
    min_repeats = _integer(min_repeats, "min_repeats", 1)
    warmup = _integer(warmup, "warmup", 0)
    max_repeats = _integer(max_repeats, "max_repeats", min_repeats)

    start = time.perf_counter()
    y = model(x)
    first = time.perf_counter() - start
    if not isinstance(y, torch.Tensor) or not bool(torch.isfinite(y).all()):
        raise ValueError("모델 출력은 유한한 값을 가진 Tensor여야 합니다.")
    for _ in range(warmup):
        model(x)

    times = []
    loop_start = time.perf_counter()
    while len(times) < max_repeats:
        start = time.perf_counter()
        y = model(x)
        times.append(time.perf_counter() - start)
        if len(times) >= min_repeats and time.perf_counter() - loop_start >= min_seconds:
            break
    wall_elapsed = time.perf_counter() - loop_start
    if not bool(torch.isfinite(y).all()):
        raise ValueError("측정 중 모델 출력에 NaN 또는 무한대가 발생했습니다.")
    repeat_count = len(times)
    actual_batch = int(x.shape[0])
    return dict(
        first_call_s=first,
        first_call_scope="이 benchmark 호출의 첫 모델 실행; 프로세스 최초·콜드 실행을 보장하지 않음",
        times_s=times, median_s=float(np.median(times)),
        p95_s=float(np.percentile(times, 95)),
        wall_elapsed_s=wall_elapsed, sum_call_s=float(sum(times)),
        repeat_count=repeat_count, actual_batch=actual_batch,
        total_images=repeat_count * actual_batch,
        warmup_count=warmup, min_seconds=float(min_seconds),
        min_repeats=min_repeats, max_repeats=max_repeats,
        measured_target_met=bool(wall_elapsed >= min_seconds and repeat_count >= min_repeats),
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(x):
    """전처리 후 텐서의 모양·자료형·바이트를 해시합니다. 타이머 밖에서 호출하세요."""
    array = x.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str((tuple(array.shape), str(array.dtype))).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def environment(model=None, x=None):
    """환경을 기록합니다. model과 x를 전달하면 CPU FP32 조건도 검증합니다."""
    if (model is None) != (x is None):
        raise ValueError("실행 검증 시 model과 x를 함께 전달하세요.")
    if model is not None:
        _validate_execution(model, x)
    board_model = None
    for path in (Path("/proc/device-tree/model"), Path("/sys/firmware/devicetree/base/model")):
        try:
            board_model = path.read_bytes().decode("utf-8", errors="replace").strip("\0\n ")
            break
        except OSError:
            pass
    record = dict(
        os=platform.platform(), architecture=platform.machine(), processor=platform.processor(),
        board_model=board_model, python=sys.version, torch=str(torch.__version__),
        numpy=np.__version__, device="CPU", dtype="FP32", execution_validated=model is not None,
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        mkldnn=bool(torch.backends.mkldnn.enabled), scope=SCOPE,
        torch_config=torch.__config__.show(), model_sha256=_sha256(ROOT / "reference.pt"),
        input_sha256=_sha256(ROOT / "sample_images.npz"), helper_sha256=_sha256(Path(__file__)),
    )
    if x is not None:
        record["input_tensor_sha256"] = tensor_sha256(x)
    return record


def compare_records(records):
    """첫 기록을 기준으로 조건을 비교하며 다른 조건과 누락을 표시합니다."""
    records = list(records)
    if not records:
        return []
    critical = ("model_sha256", "input_sha256", "dtype", "scope")
    metadata = ("torch", "numpy", "python", "threads", "interop_threads", "mkldnn", "torch_config")
    hardware = ("os", "architecture", "processor", "board_model")

    def inspect(record):
        reasons, warnings = [], []
        if not isinstance(record, dict):
            return {}, ["기록이 dict 형식이 아닙니다."], []
        env = record.get("environment")
        if not isinstance(env, dict):
            env = {}
        if record.get("schema_version") != 2:
            reasons.append("schema_version=2 기록이 아님")
        for key in critical:
            if not isinstance(env.get(key), str) or not env[key].strip():
                reasons.append(f"필수 조건 누락: {key}")
        for key, low, high in (("size", 16, None), ("batch", 1, 32)):
            try:
                _integer(record.get(key), key, low, high)
            except ValueError:
                reasons.append(f"잘못되거나 누락된 조건: {key}")
        if env.get("device") != "CPU" or env.get("dtype") != "FP32":
            reasons.append("CPU FP32 측정이 아님")
        if env.get("execution_validated") is not True:
            reasons.append("모델·입력의 CPU FP32 조건 검증 없음")
        if not record.get("conditions"):
            warnings.append("전원·냉각·백그라운드 작업 조건 기록 없음")
        measure = record.get("measurement", {})
        if not isinstance(measure, dict) or measure.get("actual_batch") != record.get("batch"):
            reasons.append("기록된 batch와 실제 측정 batch가 다르거나 누락됨")
        if isinstance(measure, dict) and measure.get("measured_target_met") is not True:
            warnings.append("최소 측정 시간·횟수 목표 미달 또는 정보 없음")
        if not env.get("input_tensor_sha256"):
            warnings.append("전처리 후 입력 텐서 해시 없음")
        for key in metadata:
            if key not in env or env[key] is None:
                warnings.append(f"환경 정보 누락: {key}")
        return env, reasons, warnings

    base = records[0]
    base_env, base_reasons, _ = inspect(base)
    rows = []
    for index, record in enumerate(records):
        env, reasons, warnings = inspect(record)
        if index:
            if base_reasons:
                reasons.append("기준 기록에 필수 조건 오류가 있음")
            if isinstance(record, dict) and isinstance(base, dict):
                for key in ("size", "batch"):
                    if record.get(key) != base.get(key):
                        reasons.append(f"기준과 다름: {key}")
                for key in critical:
                    if env.get(key) != base_env.get(key):
                        reasons.append(f"기준과 다름: {key}")
                if (env.get("input_tensor_sha256") and base_env.get("input_tensor_sha256")
                        and env["input_tensor_sha256"] != base_env["input_tensor_sha256"]):
                    reasons.append("기준과 다름: 전처리 후 실제 입력 텐서")
                if not base_env.get("input_tensor_sha256"):
                    warnings.append("기준 기록의 입력 텐서 해시 없음")
                for key in metadata + hardware + ("helper_sha256",):
                    if env.get(key) != base_env.get(key):
                        warnings.append(f"기준과 환경 차이: {key}")
                if record.get("conditions") != base.get("conditions"):
                    warnings.append("기준과 전원·냉각 등 조건 기록이 다름")
                current_measure = record.get("measurement", {})
                base_measure = base.get("measurement", {})
                if isinstance(current_measure, dict) and isinstance(base_measure, dict):
                    for key in ("warmup_count", "min_seconds", "min_repeats", "max_repeats"):
                        if current_measure.get(key) != base_measure.get(key):
                            warnings.append(f"기준과 측정 설정 차이: {key}")
            if env.get("helper_sha256") is None or base_env.get("helper_sha256") is None:
                warnings.append("제공 함수 버전 해시 누락")
        status = "비교 불가" if reasons else ("주의" if warnings else ("기준" if index == 0 else "동일 조건"))
        rows.append(dict(index=index, label=record.get("label", f"기록 {index + 1}") if isinstance(record, dict)
                         else f"기록 {index + 1}", status=status, comparable=not reasons,
                         reasons=list(dict.fromkeys(reasons)), warnings=list(dict.fromkeys(warnings))))
    return rows


def save_record(record):
    """results 안에 기존 기록을 덮어쓰지 않는 UTF-8 JSON을 저장합니다."""
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = out / f"cpu_{stamp}_{uuid.uuid4().hex[:8]}.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return path


def start():
    """노트북 공통 준비: 자료 위치 확인과 CPU 스레드 설정."""
    if ROOT != Path.cwd().resolve():
        raise RuntimeError("실습 폴더를 열고 커널을 다시 시작하세요.")
    for name in ("reference.pt", "sample_images.npz"):
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"{name} 파일이 없습니다. ZIP 전체를 압축 해제하세요.")
    torch.set_num_threads(4)


def table(headers, rows):
    """노트북에 결과 표를 표시합니다."""
    from html import escape
    from IPython.display import display, HTML
    head = "".join("<th>" + escape(str(x)) + "</th>" for x in headers)
    body = "".join("<tr>" + "".join("<td>" + escape(str(x)) + "</td>" for x in row) + "</tr>" for row in rows)
    display(HTML("<table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table>"))


def run_experiment(size, batch, label, conditions):
    """04의 제공 절차: 연산 집계, 준비 실행, 측정, 결과 저장."""
    torch.set_num_threads(4)
    x = inputs(size, batch)
    actual = int(x.shape[0])
    ops = sum(row["ops"] for row in count_ops(load(), x))
    model = load()
    measurement = benchmark(model, x, min_seconds=2.0, min_repeats=30)
    count = measurement["repeat_count"]
    elapsed = measurement["wall_elapsed_s"]
    gops = count * ops / elapsed / 1e9
    rates = dict(images_s=count * actual / elapsed, gops=gops,
                 gflops_fp32=gops, tops=gops / 1000)
    record = dict(schema_version=2, label=label, conditions=conditions,
                  environment=environment(model, x), size=size, batch=actual,
                  ops_batch=ops, measurement=measurement, rates=rates)
    save_record(record)
    return record


def list_records():
    """저장된 측정 파일을 번호와 함께 표시합니다."""
    files = sorted((ROOT / "results").glob("*.json"))
    rows = []
    for index, path in enumerate(files):
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append([index, record["label"], record["size"], record["batch"], path.name])
    table(["번호", "장치", "입력 크기", "배치", "파일"], rows)
    return files


def show_comparison(files, indices):
    """조건이 맞는 측정의 속도와 실행 환경을 보여줍니다."""
    if len(indices) < 2:
        print("비교할 두 장치의 측정 파일 번호를 입력하세요.")
        return
    for index in indices:
        _integer(index, "파일 번호", 0, len(files) - 1)
    records = [json.loads(files[index].read_text(encoding="utf-8")) for index in indices]
    checks = compare_records(records)
    rows = []
    for record, check in zip(records, checks):
        if not check["comparable"]:
            print(record["label"] + ": 측정 조건을 맞추세요. " + "; ".join(check["reasons"]))
            continue
        env = record["environment"]
        rows.append([record["label"], round(record["measurement"]["median_s"] * 1000, 3),
                     round(record["rates"]["images_s"], 2), round(record["rates"]["gflops_fp32"], 4),
                     env["torch"], env["threads"], record["conditions"]])
        differences = [w for w in check["warnings"] if w.startswith("기준과 환경 차이:")
                       or w == "기준과 전원·냉각 등 조건 기록이 다름"]
        other_notes = [w for w in check["warnings"] if w not in differences]
        if differences:
            print(record["label"] + ": 장치와 실행 환경의 차이가 함께 반영된 결과입니다.")
        if other_notes:
            print(record["label"] + ": " + "; ".join(other_notes))
    table(["장치", "호출시간 ms", "처리량 장/s", "실효 GFLOPS", "PyTorch", "스레드", "실행 조건"], rows)
