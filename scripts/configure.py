#!/usr/bin/env python3
"""Create local configurations with explicit artifact paths; never start a run."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(value, artifact_root, output_dir):
    if isinstance(value, dict):
        return {key: render(item, artifact_root, output_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [render(item, artifact_root, output_dir) for item in value]
    if isinstance(value, str):
        value = value.replace('${DPNN_ARTIFACT_ROOT}', str(artifact_root))
        if value.startswith('configs/'):
            value = str(output_dir / value.removeprefix('configs/'))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True,
                        help='Local directory containing the artifact IDs listed in configs/artifacts.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'local/configs')
    parser.add_argument('--check-artifacts', action='store_true',
                        help='Require every external path referenced by the generated execution templates to exist')
    args = parser.parse_args()
    artifact_root = args.artifact_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    templates = ROOT / 'configs'
    if output == templates or templates in output.parents:
        parser.error('Choose a local output directory outside the versioned configs/ templates')
    prepared = {}
    references = set()

    def inspect(value):
        if isinstance(value, dict):
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
        elif isinstance(value, str) and value.startswith(str(artifact_root) + '/'):
            references.add(value)

    for group in ('study', 'preparation', 'tests', 'diagnostics'):
        for source in sorted((templates / group).glob('*.json')):
            value = render(json.loads(source.read_text()), artifact_root, output)
            inspect(value)
            prepared[output / source.relative_to(templates)] = value
    missing = sorted(path for path in references if not Path(path).exists())
    if args.check_artifacts and missing:
        parser.error('Missing artifacts: ' + ', '.join(missing))
    existing = [str(path) for path in prepared if path.exists()]
    if existing:
        parser.error('Refusing to overwrite existing configurations; choose a new --output-dir')
    for target, value in prepared.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'output_directory': str(output), 'configurations_written': len(prepared),
                      'missing_artifact_paths': missing,
                      'note': 'Configurations prepared only; no experiment executed. Existing scientific hashes remain enforced.'}, indent=2))


if __name__ == '__main__':
    main()
