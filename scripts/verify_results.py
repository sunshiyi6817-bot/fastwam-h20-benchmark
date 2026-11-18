import argparse
import json
import math
import pathlib


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=pathlib.Path)
    args = parser.parse_args()
    result = load(args.run / "result.json")
    status = load(args.run / "status.json")
    require(status["exit_code"] == 0, "Training did not exit successfully")
    require(result["valid"] is True, "Result is invalid")
    require((result["warmup"], result["steps"], result["samples"]) == (10, 90, 5760), "Unexpected measurement window")
    windows = []
    for rank in range(8):
        rows = load(args.run / f"steps.rank{rank}.json")
        require([row["step"] for row in rows] == list(range(1, 101)), f"Invalid steps for rank {rank}")
        for row in rows:
            require(row["finite"] is True, f"Nonfinite flag for rank {rank}")
            require(row["samples"] == 8, f"Unexpected batch for rank {rank}")
            require(all(math.isfinite(row[key]) for key in ["loss", "grad_norm", "lr"]), f"Nonfinite value for rank {rank}")
        actual = load(args.run / f"actual.rank{rank}.json")
        for key, value in {"world": 8, "actual_micro_batch": 8, "actual_accumulation": 1, "actual_global_batch": 64, "actual_zero": 1, "actual_bf16": True}.items():
            require(actual[key] == value, f"Unexpected {key} for rank {rank}")
        window = load(args.run / f"window.rank{rank}.json")
        require(window["rank"] == rank and window["measured_steps"] == 90 and window["samples"] == 720, f"Invalid window for rank {rank}")
        require(math.isfinite(window["seconds"]) and window["seconds"] > 0, "Invalid duration")
        windows.append(window)
    duration = max(window["seconds"] for window in windows)
    samples = sum(window["samples"] for window in windows)
    require(math.isclose(duration, result["window_s"], rel_tol=1e-9), "Duration mismatch")
    require(math.isclose(samples / duration, result["samples_per_s"], rel_tol=1e-9), "Throughput mismatch")
    require(math.isclose(duration / 90, result["s_per_optimizer_step"], rel_tol=1e-9), "Step time mismatch")
    print(json.dumps({"verified_rank_steps": 800, "samples": samples, "seconds": duration, "samples_per_s": samples / duration}, indent=2))


if __name__ == "__main__":
    main()
