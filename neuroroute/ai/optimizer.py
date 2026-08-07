"""
Inference optimizer and latency profiling benchmark suite for NeuroRoute AI agents.
"""

import time
from typing import Any, Dict, Optional, Union
import numpy as np


class InferenceProfiler:
    """
    Benchmarking utility for measuring model decision latencies, percentiles, and throughput.
    """

    @staticmethod
    def profile_inference_latency(
        agent: Any,
        state_dim: int,
        num_samples: int = 1000,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Benchmark single-state and batch-state decision latencies in microseconds (us).

        Args:
            agent: DQNAgent or optimized agent instance.
            state_dim: Dimension of environment state vector.
            num_samples: Number of single-state decision samples to profile.
            batch_size: Minibatch size for batch inference profiling.

        Returns:
            Dictionary containing metrics: single_mean_us, single_p95_us, single_p99_us,
            batch_mean_us, throughput_qps, numpy_fastpath_mean_us, etc.
        """
        # Ensure agent is in evaluation mode / zero exploration for consistent profiling
        orig_epsilon = getattr(agent, "epsilon", 0.0)
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0

        # Generate random state data
        states_single = [
            np.random.randn(state_dim).astype(np.float32) for _ in range(num_samples)
        ]

        # Warmup phase (50 iterations)
        for i in range(min(50, num_samples)):
            agent.choose_action(states_single[i])

        # 1. Single State Inference Profiling
        single_latencies_us = []
        for state in states_single:
            t0 = time.perf_counter_ns()
            agent.choose_action(state)
            t1 = time.perf_counter_ns()
            single_latencies_us.append((t1 - t0) / 1000.0)

        # 2. Batch State Inference Profiling
        num_batches = max(1, num_samples // batch_size)
        batch_states = np.random.randn(num_batches, batch_size, state_dim).astype(np.float32)

        batch_latencies_us = []
        total_batch_ns = 0

        for b in range(num_batches):
            t0 = time.perf_counter_ns()
            agent.choose_action_batch(batch_states[b])
            t1 = time.perf_counter_ns()
            dt_ns = t1 - t0
            total_batch_ns += dt_ns
            batch_latencies_us.append(dt_ns / 1000.0)

        total_batch_decisions = num_batches * batch_size
        throughput_qps = (
            (total_batch_decisions / (total_batch_ns / 1e9)) if total_batch_ns > 0 else 0.0
        )

        # 3. Pure NumPy Fast-Path Profiling
        numpy_fastpath_mean_us = 0.0
        numpy_fastpath_p95_us = 0.0

        if hasattr(agent, "export_to_numpy_fastpath"):
            fastpath = agent.export_to_numpy_fastpath()
            # Warmup
            for i in range(min(50, num_samples)):
                fastpath.choose_action(states_single[i])

            numpy_latencies_us = []
            for state in states_single:
                t0 = time.perf_counter_ns()
                fastpath.choose_action(state)
                t1 = time.perf_counter_ns()
                numpy_latencies_us.append((t1 - t0) / 1000.0)

            numpy_fastpath_mean_us = float(np.mean(numpy_latencies_us))
            numpy_fastpath_p95_us = float(np.percentile(numpy_latencies_us, 95))

        # Restore original epsilon
        if hasattr(agent, "epsilon"):
            agent.epsilon = orig_epsilon

        return {
            "num_samples": num_samples,
            "single_mean_us": float(np.mean(single_latencies_us)),
            "single_p95_us": float(np.percentile(single_latencies_us, 95)),
            "single_p99_us": float(np.percentile(single_latencies_us, 99)),
            "batch_mean_us": float(np.mean(batch_latencies_us)),
            "batch_p95_us": float(np.percentile(batch_latencies_us, 95)),
            "batch_p99_us": float(np.percentile(batch_latencies_us, 99)),
            "throughput_qps": float(throughput_qps),
            "numpy_fastpath_mean_us": numpy_fastpath_mean_us,
            "numpy_fastpath_p95_us": numpy_fastpath_p95_us,
        }
