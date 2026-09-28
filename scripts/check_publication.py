"""Read-only checks for the reviewed publication working tree and Git index.

No network calls, no changes, no history rewrite, and no copyright clearance.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tif', '.tiff',
    '.pth', '.pt', '.onnx', '.ckpt', '.safetensors', '.pkl', '.pickle',
    '.zip', '.7z', '.rar', '.tar', '.gz', '.pem', '.key', '.mp4', '.npy', '.npz',
}
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.json', '.csv', '.log', '.html', '.css', '.js', '.cmd'}
TOKEN_PATTERNS = [
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    re.compile(r'\bgithub_pat_[A-Za-z0-9_]{40,}\b'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
]


def git_output(*args):
    result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)
    if result.returncode:
        raise RuntimeError('Git inspection failed: ' + ' '.join(args))
    return result.stdout


def exact_allowlist():
    # This repository deliberately uses exact file exceptions, not glob exceptions.
    lines = (ROOT / '.gitignore').read_text(encoding='utf-8-sig').splitlines()
    return {line[2:] for line in lines if line.startswith('!/')
            and not line.endswith('/') and not any(c in line for c in '*?[')}


def main():
    problems = []
    allowed = exact_allowlist()
    paths = []
    def walk(directory):
        for path in directory.iterdir():
            rel = path.relative_to(ROOT).as_posix()
            if directory == ROOT and path.name == '.git':
                if path.is_symlink():
                    problems.append('.git is a symbolic link; inspect manually')
                continue
            # Junctions/reparse points can point outside the intended directory.
            if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
                problems.append('Link/reparse point: ' + rel)
                continue
            if path.name == '.git':
                problems.append('Nested Git metadata: ' + rel)
                continue
            if path.is_dir():
                walk(path)
            elif path.is_file():
                paths.append(path)
    walk(ROOT)
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if rel not in allowed:
            # Local runtime outputs may legitimately be ignored. Conservative review.
            problems.append('Outside reviewed file allowlist: ' + rel)
        if path.suffix.lower() in BLOCKED_SUFFIXES or path.name.startswith('.env'):
            problems.append('Excluded material type: ' + rel)
        if path.stat().st_size > 20 * 1024 * 1024:
            problems.append('Large file needs manual review: ' + rel)
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == '.gitignore':
            content = path.read_text(encoding='utf-8-sig', errors='replace')
            for number, line in enumerate(content.splitlines(), 1):
                if any(pattern.search(line) for pattern in TOKEN_PATTERNS):
                    problems.append(f'Possible credential: {rel}:{number} (value hidden)')
        if path.suffix.lower() == '.docx':
            with zipfile.ZipFile(path) as doc:
                names = doc.namelist()
                if any(n.startswith(('word/media/', 'word/embeddings/'))
                       or n.endswith('vbaProject.bin') or n == 'word/comments.xml' for n in names):
                    problems.append('Word images/attachments/macros/comments: ' + rel)
                if 'docProps/core.xml' in names:
                    core = ET.fromstring(doc.read('docProps/core.xml'))
                    for node in core:
                        if node.tag.rsplit('}', 1)[-1] in {'creator', 'lastModifiedBy'} and (node.text or '').strip():
                            problems.append('Word author metadata needs manual review: ' + rel)
                for name in names:
                    if name.endswith('.rels'):
                        tree = ET.fromstring(doc.read(name))
                        if any(n.attrib.get('TargetMode') == 'External' for n in tree):
                            problems.append('Word external relationship needs manual review: ' + rel)
                            break
    manifest = json.loads((ROOT / 'experiments/release_manifest.json').read_text(encoding='utf-8-sig'))
    for item in manifest['files']:
        rel = PurePosixPath(item['path'])
        if rel.is_absolute() or '..' in rel.parts or '\\' in str(rel) or ':' in str(rel):
            problems.append('Unsafe manifest path')
            continue
        path = ROOT / 'experiments' / str(rel)
        if not path.is_file():
            problems.append('Manifest file missing: experiments/' + str(rel))
        else:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item['sha256'] or len(raw) != item['bytes']:
                problems.append('Manifest changed: experiments/' + str(rel))
    if (ROOT / '.git').exists():
        if not shutil.which('git'):
            problems.append('Git is not on PATH; index inspection unavailable')
        else:
            for raw in git_output('ls-files', '-z').split(b'\0'):
                if raw:
                    rel = raw.decode('utf-8')
                    if rel not in allowed:
                        problems.append('Tracked file outside reviewed allowlist: ' + rel)
            # Inspect staged blobs too: they can differ from current working files.
            for raw in git_output('diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z').split(b'\0'):
                if not raw:
                    continue
                rel = raw.decode('utf-8')
                current = ROOT / rel
                blob = git_output('show', ':' + rel)
                if not current.is_file() or blob != current.read_bytes():
                    problems.append('Staged file differs from checked working file; review/add again: ' + rel)
    else:
        print('No local .git: checked working files only.')
    print(f'Inspected {len(paths)} working files and {len(manifest["files"])} manifest entries.')
    if problems:
        for issue in sorted(set(problems)):
            print('REVIEW: ' + issue)
        print('Checks require attention. No files were modified.')
        return 1
    print('No covered technical issues found. Licensing/provenance and full history still require human review.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile, ET.ParseError) as error:
        print('Check could not complete: ' + str(error), file=sys.stderr)
        sys.exit(2)
