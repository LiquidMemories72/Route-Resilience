import argparse
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def _python_executable() -> str:
    python_exe = Path("venv") / "Scripts" / "python.exe"
    return str(python_exe) if python_exe.exists() else "python"


def _summary_line(summary_path: Path) -> str:
    if not summary_path.exists():
        return "summary unavailable"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "summary unreadable"

    diagnostics: Dict = summary.get("simplification_diagnostics", {}) or {}
    after = diagnostics.get("after", {})
    before = diagnostics.get("before", {})
    node_count = after.get("node_count", summary.get("nodes", "?"))
    edge_count = after.get("edge_count", summary.get("edges", "?"))
    if before:
        return (
            f"nodes {before.get('node_count', '?')} -> {node_count}, "
            f"edges {before.get('edge_count', '?')} -> {edge_count}, "
            f"deg2 removed={diagnostics.get('removed_degree2_nodes', 0)}, "
            f"short edges={diagnostics.get('removed_short_edges', 0)}, "
            f"tiny cycles={diagnostics.get('removed_tiny_cycles', 0)}, "
            f"dupes={diagnostics.get('removed_duplicate_paths', 0)}"
        )
    return f"nodes={node_count}, edges={edge_count}, components={summary.get('connected_components', '?')}"


def _run_pipeline(cmd: List[str], output_dir: Path) -> bool:
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        print("  Failed! Error output:")
        if error.stdout:
            print(error.stdout.rstrip())
        if error.stderr:
            print(error.stderr.rstrip())
        return False

    for line in result.stdout.splitlines():
        if line.startswith("Saved ") or line.startswith("[TopologyRepair]"):
            print(f"    {line}")
    print(f"    {_summary_line(output_dir / 'graph_summary.json')}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run graph extraction on random prediction masks.")
    parser.add_argument("--mask-dir", default="pred_masks")
    parser.add_argument("--sat-dir", default="valid")
    parser.add_argument("--output-dir", default="graph_random_samples")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--compare-no-simplify", action="store_true")
    parser.add_argument("--no-repair", action="store_true", help="Do not pass probability maps to topology repair.")
    parser.add_argument("--no-simplify", action="store_true", help="Disable graph simplification for the main run.")
    parser.add_argument("--no-contract-degree2", action="store_true")
    parser.add_argument("--no-collapse-short-edges", action="store_true")
    parser.add_argument("--no-deduplicate-paths", action="store_true")
    parser.add_argument("--no-remove-tiny-cycles", action="store_true")
    parser.add_argument("--short-edge-threshold", type=float, default=4.0)
    parser.add_argument("--tiny-cycle-perimeter", type=float, default=18.0)
    parser.add_argument("--tiny-cycle-radius", type=float, default=4.0)
    parser.add_argument("--max-artifact-edge-length", type=float, default=8.0)
    return parser


def _base_command(args: argparse.Namespace, mask_path: Path, output_dir: Path) -> List[str]:
    base_id = mask_path.name.replace("_pred.png", "")
    prob_path = Path(args.mask_dir) / f"{base_id}_prob.npy"
    sat_path = Path(args.sat_dir) / f"{base_id}_sat.jpg"

    cmd = [_python_executable(), str(Path("graph_module") / "run_pipeline.py")]
    cmd.extend(["--mask", str(mask_path)])

    if not args.no_repair and prob_path.exists():
        cmd.extend(["--prob-map", str(prob_path)])
    elif not args.no_repair:
        print(f"  Warning: No probability map found at {prob_path}")

    if sat_path.exists():
        cmd.extend(["--satellite", str(sat_path)])
    else:
        print(f"  Warning: No satellite image found at {sat_path}")

    cmd.extend(["--output-dir", str(output_dir)])
    cmd.extend(["--short-edge-threshold", str(args.short_edge_threshold)])
    cmd.extend(["--tiny-cycle-perimeter", str(args.tiny_cycle_perimeter)])
    cmd.extend(["--tiny-cycle-radius", str(args.tiny_cycle_radius)])
    cmd.extend(["--max-artifact-edge-length", str(args.max_artifact_edge_length)])

    if args.no_simplify:
        cmd.append("--no-simplify")
    if args.no_contract_degree2:
        cmd.append("--no-contract-degree2")
    if args.no_collapse_short_edges:
        cmd.append("--no-collapse-short-edges")
    if args.no_deduplicate_paths:
        cmd.append("--no-deduplicate-paths")
    if args.no_remove_tiny_cycles:
        cmd.append("--no-remove-tiny-cycles")
    return cmd


def main() -> None:
    args = build_parser().parse_args()
    mask_dir = Path(args.mask_dir)
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(exist_ok=True)

    all_masks = sorted(mask_dir.glob("*_pred.png"))
    if not all_masks:
        print(f"No masks found in {mask_dir}")
        return

    if args.seed is not None:
        random.seed(args.seed)

    num_samples = min(args.samples, len(all_masks))
    random_masks = random.sample(all_masks, num_samples)
    print(f"Selected {num_samples} random images to process.\n")

    for index, mask_path in enumerate(random_masks, 1):
        base_id = mask_path.name.replace("_pred.png", "")
        output_dir = output_base_dir / base_id
        print(f"[{index}/{num_samples}] Processing image ID: {base_id}")

        cmd = _base_command(args, mask_path, output_dir)
        print("  Simplified run:")
        ok = _run_pipeline(cmd, output_dir)

        if ok and args.compare_no_simplify and not args.no_simplify:
            baseline_dir = output_base_dir / f"{base_id}_nosimplify"
            baseline_cmd = _base_command(args, mask_path, baseline_dir)
            baseline_cmd.append("--no-simplify")
            print("  No-simplify comparison run:")
            _run_pipeline(baseline_cmd, baseline_dir)

        print("-" * 40)

    print(f"\nAll done! Check outputs inside '{output_base_dir}'.")


if __name__ == "__main__":
    main()
