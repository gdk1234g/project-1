"""Create a unique offline Git practice repository in the system temp directory.

Never changes this project's Git settings, never pushes, never deletes old labs.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    git = shutil.which('git')
    if git is None:
        raise SystemExit('Install Git for Windows and reopen the terminal before using this script.')
    lab = Path(tempfile.mkdtemp(prefix='git-learning-'))
    print('New independent practice directory: ' + str(lab), flush=True)

    def run(*args):
        print('git ' + ' '.join(args), flush=True)
        subprocess.run([git, '-C', str(lab), *args], check=True)

    run('init', '-b', 'main')
    run('config', '--local', 'user.name', 'Git Learner')
    run('config', '--local', 'user.email', 'learner@example.invalid')
    run('config', '--local', 'core.autocrlf', 'false')
    # Avoid inheriting global signing or hook behavior for this isolated exercise.
    run('config', '--local', 'commit.gpgsign', 'false')
    run('config', '--local', 'core.hooksPath', '.git/learning-empty-hooks')
    (lab / '.git/learning-empty-hooks').mkdir(exist_ok=True)
    (lab / 'notes.txt').write_text('My first Git note.\n', encoding='utf-8')
    (lab / 'message.txt').write_text('Original shared message.\n', encoding='utf-8')
    run('add', '--', 'notes.txt', 'message.txt')
    run('commit', '-m', 'docs: start independent Git learning lab')
    print('Copy the next command into PowerShell:')
    print("Set-Location -LiteralPath '" + str(lab).replace("'", "''") + "'")
    print('No remote configured. Follow your separately saved GIT_EXERCISES.md guide.')


if __name__ == '__main__':
    main()
