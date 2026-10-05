"""Run a trusted user module's factory. No hidden model downloads or installations."""
import argparse
import importlib
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hallulab")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Package metadata, or explicit runtime preflight")
    doctor.add_argument("--runtime", choices=("af2", "af3"),
                        help="Import runtime and initialize JAX devices; no model execution")
    doctor.add_argument("--require-gpu", action="store_true")
    doctor.add_argument("--weights", type=Path, help="Check local weight-file presence only")
    doctor.add_argument("--profile", type=Path, help="Check Python and exact package versions")
    run_parser = sub.add_parser("run", help="Execute trusted module:function(config) -> Run")
    run_parser.add_argument("factory", help="Importable module:function; executes trusted Python code")
    run_parser.add_argument("--config", type=Path, help="JSON passed to factory (default: {})")
    run_parser.add_argument("--resume", type=Path)
    run_parser.add_argument("--max-steps", type=int)
    run_parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .diagnostics import preflight
            result = preflight(runtime=args.runtime, require_gpu=args.require_gpu,
                               weights=args.weights, profile=args.profile)
            print(json.dumps(result, indent=2))
            if not result["ok"]:
                parser.exit(2)
            return
        parts = args.factory.split(":")
        if len(parts) != 2 or not all(parts):
            raise ValueError("factory must use module:function format")
        module, name = parts
        imported = importlib.import_module(module)
        try:
            factory = getattr(imported, name)
        except AttributeError:
            raise ValueError(f"Factory {name!r} not found in module {module!r}") from None
        if not callable(factory):
            raise ValueError("Factory must be callable")
        config = json.loads(args.config.read_text()) if args.config else {}
        if not isinstance(config, dict):
            raise ValueError("config must be a JSON object")
        from .engine import Run
        run = factory(config)
        if not isinstance(run, Run):
            raise TypeError("factory must return a hallucination_lab.Run")
        if args.resume:
            run.restore(args.resume)
        run.run(max_steps=args.max_steps, max_seconds=args.max_seconds)
        print(json.dumps({"completed_steps": run.state.step, "total_steps": run.total_steps,
                          "best_evaluated_loss": run.state.best_loss}))
    except (ValueError, TypeError, ImportError, OSError, FloatingPointError) as error:
        evidence = getattr(error, "evidence_directory", None)
        detail = f"; evidence: {evidence}" if evidence else ""
        parser.exit(2, f"hallulab: {error}{detail}\n")


if __name__ == "__main__":
    main()
