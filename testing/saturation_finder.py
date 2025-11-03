"""
DeepSeek-VL2 saturation estimator (vLLM / SGLang)

- Polls /metrics (Prometheus text) and samples:
  * num_running_reqs
  * num_queue_reqs
  * optional token/cache usage

- Recommends a queue length that keeps the engine saturated:
  p95(queue length) during periods where num_running_reqs >= 95% of observed max
  (configurable via --util-threshold), with a small +1 headroom.

Usage examples:
  python saturation_finder.py --backend vllm   --base-url http://localhost:8000 --model deepseek-vl2
  python saturation_finder.py --backend sglang --base-url http://localhost:30000 --model deepseek-vl2

Requires: pip install aiohttp
"""

import argparse
import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


# ----------------------------
# Data classes
# ----------------------------

@dataclass
class QueueMetricsSample:
    """A single sample of queue metrics at a point in time."""
    timestamp: float
    num_queue_reqs: int
    avg_queue_latency: float
    num_running_reqs: int
    token_usage: float


@dataclass
class ServerMetrics:
    """Server-side metrics from Prometheus."""
    num_requests: int
    num_aborted_requests: int
    prompt_tokens: int
    generation_tokens: int
    cached_tokens: int

    prefill_throughput_buckets: Dict[float, float] = field(default_factory=dict)
    decode_throughput_buckets: Dict[float, float] = field(default_factory=dict)

    prefill_throughput_sum: float = 0.0
    decode_throughput_sum: float = 0.0

    avg_batch_size: float = 0.0

    num_queue_reqs: int = 0
    avg_request_queue_latency: float = 0.0

    queue_length_buckets: Dict[float, float] = field(default_factory=dict)
    queue_latency_buckets: Dict[float, float] = field(default_factory=dict)

    queue_length_sum: float = 0.0
    queue_latency_sum: float = 0.0


# ----------------------------
# Prometheus parsing utilities
# ----------------------------

def parse_prometheus_metrics(
    text: str
) -> Tuple[Dict[str, float], Dict[str, Dict[float, float]], Dict[str, Dict[tuple, float]]]:
    """Parse Prometheus text format metrics into:
       - metrics: flat {name -> value} (labelled metrics are summed by name)
       - histograms: {base_metric_name -> {le -> cumulative_count}}
       - labeled_metrics: {name -> {tuple(sorted(label_items)) -> value}}
    """
    metrics: Dict[str, float] = {}
    histograms: Dict[str, Dict[float, float]] = {}
    labeled_metrics: Dict[str, Dict[tuple, float]] = {}

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Histogram buckets: <metric>_bucket{... le="<bound>"} <count>
        bucket_match = re.match(
            r'^([a-zA-Z_:][a-zA-Z0-9_:]*_bucket)\{.*?le="([\d.e+-]+|\\\+Inf)".*?\}\s+([\d.e+-]+)$',
            line,
        )
        if bucket_match:
            metric_name = bucket_match.group(1).replace("_bucket", "")
            le_value = bucket_match.group(2)
            count = float(bucket_match.group(3))

            if le_value in ["+Inf", r"\+Inf"]:
                le_num = float("inf")
            else:
                le_num = float(le_value)

            histograms.setdefault(metric_name, {})[le_num] = count
            continue

        # Labeled metrics: <metric>{k="v",...} <value>
        label_match = re.match(
            r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{(.*?)\}\s+([\d.e+-]+)$',
            line,
        )
        if label_match:
            metric_name = label_match.group(1)
            labels_str = label_match.group(2)
            value = float(label_match.group(3))

            labels: Dict[str, str] = {}
            for label_pair in labels_str.split(","):
                if "=" in label_pair:
                    k, v = label_pair.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')

            label_key = tuple(sorted(labels.items()))
            labeled_metrics.setdefault(metric_name, {})[label_key] = value
            metrics[metric_name] = metrics.get(metric_name, 0.0) + value
            continue

        # Plain metrics: <metric> <value>
        plain_match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([\d.e+-]+)$", line)
        if plain_match:
            metric_name = plain_match.group(1)
            value = float(plain_match.group(2))
            metrics[metric_name] = value

    return metrics, histograms, labeled_metrics


def get_labeled_metric_value(
    labeled_metrics: Dict[str, Dict[tuple, float]],
    metric_name: str,
    label_filters: Dict[str, str],
) -> float:
    """Sum metric values that match all label_filters."""
    if metric_name not in labeled_metrics:
        return 0.0

    total = 0.0
    for label_tuple, value in labeled_metrics[metric_name].items():
        labels_dict = dict(label_tuple)
        if all(labels_dict.get(k) == v for k, v in label_filters.items()):
            total += value
    return total


def calculate_percentiles_from_histogram(
    buckets: Dict[float, float],
    percentiles: List[float],
    first_lower_bound: Optional[float] = None,
) -> Dict[float, float]:
    """Percentiles from Prom cumulative buckets using linear interpolation."""
    if not buckets:
        return {float(p): 0.0 for p in percentiles}

    sorted_buckets = sorted(buckets.items())
    uppers, cums = zip(*sorted_buckets)

    # Sanity: cumulative must be non-decreasing
    for i in range(1, len(cums)):
        if cums[i] < cums[i - 1]:
            raise ValueError("Cumulative counts must be non-decreasing.")

    total = cums[-1]
    if total <= 0:
        return {float(p): 0.0 for p in percentiles}

    first_lower = uppers[0] if first_lower_bound is None else first_lower_bound

    results: Dict[float, float] = {}
    for p in percentiles:
        pp = max(0.0, min(100.0, float(p)))
        if pp == 0.0:
            results[p] = float(first_lower)
            continue
        if pp == 100.0:
            results[p] = float(uppers[-1])
            continue

        target = (pp / 100.0) * total
        prev_upper = first_lower
        prev_cum = 0.0

        for upper, cum in sorted_buckets:
            if cum >= target:
                if cum == prev_cum:
                    results[p] = float(upper)
                else:
                    frac = (target - prev_cum) / (cum - prev_cum)
                    results[p] = float(prev_upper + frac * (upper - prev_upper))
                break
            prev_upper, prev_cum = upper, cum

    return results


# ----------------------------
# Metrics fetchers
# ----------------------------

async def fetch_server_metrics(base_url: str, backend: str = "luminal") -> Optional[ServerMetrics]:
    """Fetch one snapshot of server-side metrics from /metrics."""
    metrics_url = f"{base_url}/metrics"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    print(f"Warning: Failed to fetch metrics from {metrics_url} (status {resp.status})")
                    return None
                text = await resp.text()
                parsed, histograms, labeled_metrics = parse_prometheus_metrics(text)

                if backend == "vllm":
                    num_requests = int(parsed.get("vllm:request_success_total", 0))
                    num_aborted = int(
                        get_labeled_metric_value(
                            labeled_metrics,
                            "vllm:request_success_total",
                            {"finish_reason": "abort"},
                        )
                    )
                    return ServerMetrics(
                        num_requests=num_requests,
                        num_aborted_requests=num_aborted,
                        prompt_tokens=int(parsed.get("vllm:prompt_tokens_total", 0)),
                        generation_tokens=int(parsed.get("vllm:generation_tokens_total", 0)),
                        cached_tokens=int(parsed.get("vllm:prefix_cache_hits", 0)),
                        prefill_throughput_buckets=histograms.get("vllm:time_to_first_token_seconds", {}),
                        decode_throughput_buckets=histograms.get("vllm:time_per_output_token_seconds", {}),
                        prefill_throughput_sum=parsed.get("vllm:time_to_first_token_seconds_sum", 0.0),
                        decode_throughput_sum=parsed.get("vllm:time_per_output_token_seconds_sum", 0.0),
                        # Note: using running reqs as a proxy; rename if desired
                        avg_batch_size=parsed.get("vllm:num_requests_running", 0.0),
                        num_queue_reqs=int(parsed.get("vllm:num_requests_waiting", 0)),
                        avg_request_queue_latency=(
                            parsed.get("vllm:request_queue_time_seconds_sum", 0.0)
                            / max(parsed.get("vllm:request_queue_time_seconds_count", 1), 1)
                        ),
                        queue_length_buckets=histograms.get("vllm:num_requests_waiting", {}),
                        queue_latency_buckets=histograms.get("vllm:request_queue_time_seconds", {}),
                        queue_length_sum=parsed.get("vllm:num_requests_waiting_sum", 0.0),
                        queue_latency_sum=parsed.get("vllm:request_queue_time_seconds_sum", 0.0),
                    )

                if backend == "sglang":
                    return ServerMetrics(
                        num_requests=int(parsed.get("sglang:num_requests_total", 0)),
                        num_aborted_requests=0,  # not found
                        prompt_tokens=int(parsed.get("sglang:prompt_tokens_total", 0)),
                        generation_tokens=int(parsed.get("sglang:generation_tokens_total", 0)),
                        cached_tokens=int(parsed.get("sglang:cached_tokens_total", 0)),
                        prefill_throughput_buckets=histograms.get("sglang:time_to_first_token_seconds", {}),
                        decode_throughput_buckets={},  # not found
                        prefill_throughput_sum=parsed.get("sglang:time_to_first_token_seconds_sum", 0.0),
                        decode_throughput_sum=0.0,  # not found
                        avg_batch_size=parsed.get("sglang:num_running_reqs", 0.0),
                        num_queue_reqs=int(parsed.get("sglang:num_queue_reqs", 0)),
                        avg_request_queue_latency=(
                            parsed.get("sglang:queue_time_seconds_sum", 0.0)
                            / max(parsed.get("sglang:queue_time_seconds_count", 1), 1)
                        ),
                        queue_length_buckets={},  # not found
                        queue_latency_buckets=histograms.get("sglang:queue_time_seconds", {}),
                        queue_length_sum=0.0,  # not found
                        queue_latency_sum=parsed.get("sglang:queue_time_seconds_sum", 0.0),
                    )

                # default "luminal" block (kept for completeness)
                return ServerMetrics(
                    num_requests=int(parsed.get("luminal:num_requests_total", 0)),
                    num_aborted_requests=int(parsed.get("luminal:num_aborted_requests_total", 0)),
                    prompt_tokens=int(parsed.get("luminal:prompt_tokens_total", 0)),
                    generation_tokens=int(parsed.get("luminal:generation_tokens_total", 0)),
                    cached_tokens=int(parsed.get("luminal:cached_tokens_total", 0)),
                    prefill_throughput_buckets=histograms.get("luminal:input_throughput_histogram", {}),
                    decode_throughput_buckets=histograms.get("luminal:gen_throughput_histogram", {}),
                    prefill_throughput_sum=parsed.get("luminal:input_throughput_histogram_sum", 0.0),
                    decode_throughput_sum=parsed.get("luminal:gen_throughput_histogram_sum", 0.0),
                    avg_batch_size=parsed.get("luminal:avg_batch_size", 0.0),
                    num_queue_reqs=int(parsed.get("luminal:num_queue_reqs", 0)),
                    avg_request_queue_latency=parsed.get("luminal:avg_request_queue_latency", 0.0),
                    queue_length_buckets=histograms.get("luminal:num_queue_reqs_histogram", {}),
                    queue_latency_buckets=histograms.get("luminal:avg_request_queue_latency_histogram", {}),
                    queue_length_sum=parsed.get("luminal:num_queue_reqs_histogram_sum", 0.0),
                    queue_latency_sum=parsed.get("luminal:avg_request_queue_latency_histogram_sum", 0.0),
                )
    except Exception as e:
        print(f"Warning: Failed to fetch server metrics: {e}")
        return None


def total_from_cumhist_delta(delta_buckets: Dict[float, float]) -> float:
    """Extract total count from cumulative histogram delta buckets."""
    if not delta_buckets:
        return 0.0
    last = -float("inf")
    for _, c in sorted(delta_buckets.items()):
        if c < last:
            # Not monotonic: treat values as raw counts and sum them
            return sum(delta_buckets.values())
        last = c
    return max(delta_buckets.values())


def calculate_server_metrics_delta(before: ServerMetrics, after: ServerMetrics, duration_s: float) -> Dict[str, Any]:
    """Delta between two snapshots (kept for completeness; not used by the estimator)."""
    delta_requests = after.num_requests - before.num_requests
    delta_aborted = after.num_aborted_requests - before.num_aborted_requests
    delta_prompt_tokens = after.prompt_tokens - before.prompt_tokens
    delta_generation_tokens = after.generation_tokens - before.generation_tokens
    delta_cached_tokens = after.cached_tokens - before.cached_tokens

    # Prefill / decode hist deltas
    dpref = {b: after.prefill_throughput_buckets.get(b, 0) - before.prefill_throughput_buckets.get(b, 0)
             for b in set(before.prefill_throughput_buckets) | set(after.prefill_throughput_buckets)}
    ddec = {b: after.decode_throughput_buckets.get(b, 0) - before.decode_throughput_buckets.get(b, 0)
            for b in set(before.decode_throughput_buckets) | set(after.decode_throughput_buckets)}

    ptiles = [50, 95, 99]
    prefill_percentiles = calculate_percentiles_from_histogram(dpref, ptiles)
    decode_percentiles = calculate_percentiles_from_histogram(ddec, ptiles)

    delta_prefill_sum = after.prefill_throughput_sum - before.prefill_throughput_sum
    delta_decode_sum = after.decode_throughput_sum - before.decode_throughput_sum

    prefill_count = total_from_cumhist_delta(dpref)
    decode_count = total_from_cumhist_delta(ddec)

    prefill_avg = delta_prefill_sum / prefill_count if prefill_count > 0 else 0.0
    decode_avg = delta_decode_sum / decode_count if decode_count > 0 else 0.0

    avg_cached_per_request = (delta_cached_tokens / delta_requests) if delta_requests > 0 else 0.0
    cache_hit_rate = (delta_cached_tokens / delta_prompt_tokens * 100) if delta_prompt_tokens > 0 else 0.0

    # Queue stats (if present)
    dq_len = {b: after.queue_length_buckets.get(b, 0) - before.queue_length_buckets.get(b, 0)
              for b in set(before.queue_length_buckets) | set(after.queue_length_buckets)}
    dq_lat = {b: after.queue_latency_buckets.get(b, 0) - before.queue_latency_buckets.get(b, 0)
              for b in set(before.queue_latency_buckets) | set(after.queue_latency_buckets)}

    q_len_percentiles = calculate_percentiles_from_histogram(dq_len, ptiles, first_lower_bound=0.0)
    q_lat_percentiles = calculate_percentiles_from_histogram(dq_lat, ptiles, first_lower_bound=0.0)

    delta_queue_length_sum = after.queue_length_sum - before.queue_length_sum
    delta_queue_latency_sum = after.queue_latency_sum - before.queue_latency_sum

    q_len_count = total_from_cumhist_delta(dq_len)
    q_lat_count = total_from_cumhist_delta(dq_lat)

    q_len_avg = delta_queue_length_sum / q_len_count if q_len_count > 0 else 0.0
    q_lat_avg_ms = (delta_queue_latency_sum / q_lat_count * 1000) if q_lat_count > 0 else 0.0

    return {
        "requests": delta_requests,
        "aborted_requests": delta_aborted,
        "successful_requests": delta_requests - delta_aborted,
        "prompt_tokens": delta_prompt_tokens,
        "generation_tokens": delta_generation_tokens,
        "cached_tokens": delta_cached_tokens,
        "prefill_throughput": {"p50": prefill_percentiles.get(50, 0.0),
                               "p95": prefill_percentiles.get(95, 0.0),
                               "p99": prefill_percentiles.get(99, 0.0),
                               "avg": prefill_avg},
        "decode_throughput": {"p50": decode_percentiles.get(50, 0.0),
                              "p95": decode_percentiles.get(95, 0.0),
                              "p99": decode_percentiles.get(99, 0.0),
                              "avg": decode_avg},
        "cached_tokens_per_request": {"p50": None, "p95": None, "p99": None, "avg": avg_cached_per_request},
        "cache_hit_rate_percent": cache_hit_rate,
        "avg_batch_size": after.avg_batch_size,
        "queue_length": {"p50": q_len_percentiles.get(50, 0.0),
                         "p95": q_len_percentiles.get(95, 0.0),
                         "p99": q_len_percentiles.get(99, 0.0),
                         "avg": q_len_avg},
        "queue_latency_ms": {"p50": q_lat_percentiles.get(50, 0.0) * 1000,
                             "p95": q_lat_percentiles.get(95, 0.0) * 1000,
                             "p99": q_lat_percentiles.get(99, 0.0) * 1000,
                             "avg": q_lat_avg_ms},
    }


# ----------------------------
# Polling sampler
# ----------------------------

async def poll_queue_metrics(
    base_url: str,
    start_time: float,
    samples: List[QueueMetricsSample],
    poll_interval: float = 5.0,
    stop_event: Optional[asyncio.Event] = None,
    backend: str = "luminal",
):
    """Poll queue metrics into `samples` until stop_event is set."""
    assert stop_event is not None, "stop_event must be provided"
    metrics_url = f"{base_url}/metrics"

    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while not stop_event.is_set():
            try:
                async with session.get(metrics_url) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        parsed, _, _ = parse_prometheus_metrics(text)

                        if backend == "vllm":
                            sample = QueueMetricsSample(
                                timestamp=time.perf_counter() - start_time,
                                num_queue_reqs=int(parsed.get("vllm:num_requests_waiting", 0)),
                                avg_queue_latency=0.0,
                                num_running_reqs=int(parsed.get("vllm:num_requests_running", 0)),
                                token_usage=float(parsed.get("vllm:gpu_cache_usage_perc", 0.0)),
                            )
                        elif backend == "sglang":
                            sample = QueueMetricsSample(
                                timestamp=time.perf_counter() - start_time,
                                num_queue_reqs=int(parsed.get("sglang:num_queue_reqs", 0)),
                                avg_queue_latency=0.0,
                                num_running_reqs=int(parsed.get("sglang:num_running_reqs", 0)),
                                token_usage=float(parsed.get("sglang:token_usage", 0.0)),
                            )
                        else:
                            sample = QueueMetricsSample(
                                timestamp=time.perf_counter() - start_time,
                                num_queue_reqs=int(parsed.get("luminal:num_queue_reqs", 0)),
                                avg_queue_latency=float(parsed.get("luminal:avg_request_queue_latency", 0.0)),
                                num_running_reqs=int(parsed.get("luminal:num_running_reqs", 0)),
                                token_usage=float(parsed.get("luminal:token_usage", 0.0)),
                            )
                        samples.append(sample)
            except Exception as e:
                print(f"Warning: Failed to poll queue metrics: {e}")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                pass


# ----------------------------
# Saturation Finder
# ----------------------------

class SaturationFinder:
    """
    Estimate the queue length needed to achieve/maintain near-full engine saturation
    for DeepSeek-VL2 on vLLM or SGLang.

    Heuristic:
      1) Observe max(num_running_reqs) over the sampling window.
      2) target_running = ceil(util_threshold * max_running).
      3) During times where running >= target, compute p95 of queue length.
      4) Recommend that p95 + 1 as steady-state queue length (>= 1).
    """

    def __init__(
        self,
        base_url: str,
        backend: str,
        duration_s: float = 60.0,
        poll_interval_s: float = 1.0,
        util_threshold: float = 0.95,
        min_samples_for_confident: int = 10,
        model_name: str = "deepseek-vl2",
    ):
        self.base_url = base_url.rstrip("/")
        self.backend = backend
        self.duration_s = duration_s
        self.poll_interval_s = poll_interval_s
        self.util_threshold = util_threshold
        self.min_samples_for_confident = min_samples_for_confident
        self.model_name = model_name

        self.samples: List[QueueMetricsSample] = []

    async def run(self) -> Dict[str, Any]:
        start = time.perf_counter()
        stop_event = asyncio.Event()

        async def _stopper():
            await asyncio.sleep(self.duration_s)
            stop_event.set()

        poll_task = asyncio.create_task(
            poll_queue_metrics(
                base_url=self.base_url,
                start_time=start,
                samples=self.samples,
                poll_interval=self.poll_interval_s,
                stop_event=stop_event,
                backend=self.backend,
            )
        )
        stopper_task = asyncio.create_task(_stopper())

        await asyncio.gather(poll_task, stopper_task)
        return self._analyze()

    def _analyze(self) -> Dict[str, Any]:
        if not self.samples:
            return {
                "backend": self.backend,
                "model": self.model_name,
                "recommended_queue_length": 0,
                "confidence": "low",
                "reason": "No samples collected.",
                "stats": {},
            }

        def percentile(values: List[float], p: float) -> float:
            if not values:
                return 0.0
            vs = sorted(values)
            if len(vs) == 1:
                return float(vs[0])
            k = (len(vs) - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return float(vs[int(k)])
            return float(vs[f] + (k - f) * (vs[c] - vs[f]))

        max_running = max(s.num_running_reqs for s in self.samples)
        target_running = max(1, math.ceil(self.util_threshold * max_running))

        saturated_samples = [s for s in self.samples if s.num_running_reqs >= target_running]

        overall_q = [s.num_queue_reqs for s in self.samples]
        sat_q = [s.num_queue_reqs for s in saturated_samples]

        overall_p50 = percentile(overall_q, 50)
        overall_p95 = percentile(overall_q, 95)
        sat_p50 = percentile(sat_q, 50) if sat_q else 0.0
        sat_p95 = percentile(sat_q, 95) if sat_q else 0.0

        if len(saturated_samples) >= self.min_samples_for_confident:
            rec = max(1, int(round(sat_p95)) + 1)
            confidence = "high"
            basis = "saturated-window p95"
            used_p95 = sat_p95
        else:
            rec = max(1, int(round(overall_p95)) + 1)
            confidence = "medium" if len(self.samples) >= self.min_samples_for_confident else "low"
            basis = "overall p95"
            used_p95 = overall_p95

        token_usages = [s.token_usage for s in self.samples if isinstance(s.token_usage, (int, float))]
        token_usage_p50 = percentile(token_usages, 50) if token_usages else None
        token_usage_p95 = percentile(token_usages, 95) if token_usages else None

        return {
            "backend": self.backend,
            "model": self.model_name,
            "recommended_queue_length": rec,
            "confidence": confidence,
            "basis": basis,
            "stats": {
                "max_running_reqs": max_running,
                "target_running_reqs": target_running,
                "samples_total": len(self.samples),
                "samples_saturated": len(saturated_samples),
                "queue_len": {
                    "overall_p50": overall_p50,
                    "overall_p95": overall_p95,
                    "saturated_p50": sat_p50,
                    "saturated_p95": sat_p95,
                    "used_p95": used_p95,
                },
                "token_usage_percent": {
                    "p50": token_usage_p50,
                    "p95": token_usage_p95,
                },
            },
        }


# ----------------------------
# CLI
# ----------------------------

async def main_async(args) -> None:
    finder = SaturationFinder(
        base_url=args.base_url,
        backend=args.backend,
        duration_s=args.duration,
        poll_interval_s=args.poll_interval,
        util_threshold=args.util_threshold,
        model_name=args.model,
    )
    result = await finder.run()

    print("\n=== DeepSeek-VL2 Saturation Estimate ===")
    print(f"Backend: {result['backend']} | Model: {result['model']}")
    print(
        f"Recommended queue length: {result['recommended_queue_length']} "
        f"(confidence: {result['confidence']}, basis: {result.get('basis')})"
    )
    stats = result["stats"]
    if stats:
        print(f"Max running reqs: {stats['max_running_reqs']} | Target running: {stats['target_running_reqs']}")
        print(f"Samples: total={stats['samples_total']} saturated={stats['samples_saturated']}")
        q = stats["queue_len"]
        print(f"Queue length p50/p95 overall: {q['overall_p50']:.2f}/{q['overall_p95']:.2f}")
        print(f"Queue length p50/p95 (saturated): {q['saturated_p50']:.2f}/{q['saturated_p95']:.2f}")
        tu = stats.get("token_usage_percent")
        if tu and (tu["p50"] is not None):
            print(f"Token usage % p50/p95: {tu['p50']:.1f}% / {tu['p95']:.1f}%")
    print("========================================\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Estimate queue length to keep DeepSeek-VL2 saturated on vLLM or SGLang."
    )
    p.add_argument("--base-url", required=True, help="Base URL to the serving instance (e.g., http://localhost:8000)")
    p.add_argument("--backend", choices=["vllm", "sglang"], required=True, help="Backend type")
    p.add_argument("--model", default="deepseek-vl2", help="Model name (for labeling only)")
    p.add_argument("--duration", type=float, default=60.0, help="Sampling duration in seconds (default: 60)")
    p.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds (default: 1)")
    p.add_argument("--util-threshold", type=float, default=0.95, help="Target running reqs fraction (default: 0.95)")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
