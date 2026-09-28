"""Preview or explicitly refresh hashes of existing reviewed record entries.

This does not include new files or grant permission to publish changed content.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Write changes after manual content review')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'experiments'
    manifest_path = root / 'release_manifest.json'
    original = manifest_path.read_bytes()
    manifest = json.loads(original.decode('utf-8-sig'))
    changes = []
    for item in manifest['files']:
        rel = PurePosixPath(item['path'])
        if rel.is_absolute() or '..' in rel.parts or '\\' in str(rel) or ':' in str(rel):
            raise ValueError('Unsafe manifest path')
        path = root / str(rel)
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Manifest path escapes experiment directory')
        raw = path.read_bytes()  # Missing files fail rather than silently disappearing.
        sha = hashlib.sha256(raw).hexdigest()
        if sha != item['sha256'] or len(raw) != item['bytes']:
            changes.append(str(rel))
            item.update(sha256=sha, bytes=len(raw))
    for rel in changes:
        print('CHANGED: ' + rel)
    if not changes:
        print('No changes.')
        return
    if not args.write:
        print('Preview only. Review content first; use --write to refresh existing entries.')
        return
    if manifest_path.read_bytes() != original:
        raise RuntimeError('Manifest changed during review; retry after checking edits')
    new_content = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=root, prefix='.manifest-', suffix='.tmp', delete=False) as temp:
            temp_name = temp.name
            temp.write(new_content)
        os.replace(temp_name, manifest_path)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    print('Refreshed existing entries only. Inspect git diff and commit changed records with this manifest.')


if __name__ == '__main__':
    main()
