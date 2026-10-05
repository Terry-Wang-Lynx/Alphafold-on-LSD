"""Verify installed ColabDesign Python files without importing its model runtime.

This checks source identity, not GPU readiness or numerical correctness. Keep the
environment unchanged after verification and retain each Run's manifest.
"""
import argparse
import hashlib
from importlib import metadata, util
import json
from pathlib import Path


def source_map(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*.py'))
            if not any(part.startswith('.') or part == '__pycache__'
                       for part in path.relative_to(root).parts)}


def compare(expected, actual):
    return {'missing': sorted(expected.keys() - actual.keys()),
            'extra': sorted(actual.keys() - expected.keys()),
            'changed': sorted(name for name in expected.keys() & actual.keys()
                              if expected[name] != actual[name])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path,
                        default=Path(__file__).with_name('af2-py312-source.json'))
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    spec = util.find_spec(reference['package'])
    if spec is None or spec.origin is None:
        parser.error('ColabDesign is not installed in this Python environment')
    root = Path(spec.origin).parent
    actual = source_map(root)
    difference = compare(reference['python_files'], actual)
    installed_version = metadata.version(reference['package'])
    passed = not any(difference.values()) and installed_version == reference['version']
    print(json.dumps({'passed': passed, 'package': reference['package'],
                      'installed_version': installed_version, 'source_root': str(root),
                      'python_file_count': len(actual), 'difference': difference,
                      'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                      'scope': 'Installed Python source identity only'}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
