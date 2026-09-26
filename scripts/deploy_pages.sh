#!/usr/bin/env bash
# Publish site/ to the gh-pages branch, served at https://4waiz.github.io/Dapper/
#
#   bash scripts/deploy_pages.sh ["commit message"]
#
# Commits are authored with your local git identity, so no bot appears in the
# history. The branch is rebuilt from site/ each time rather than merged, so a
# file deleted from site/ really disappears from the published site.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MESSAGE="${1:-deploy: publish the dashboard and the paper page}"
WORK="$(mktemp -d)"
trap 'git -C "$ROOT" worktree remove --force "$WORK" >/dev/null 2>&1 || true' EXIT

cd "$ROOT"

if [ ! -f site/index.html ] || [ ! -f site/data/config.json ]; then
  echo "site/ is missing its entry point or its exported data; run the exporters first" >&2
  exit 1
fi

git fetch origin gh-pages >/dev/null 2>&1 || true
if git show-ref --verify --quiet refs/remotes/origin/gh-pages; then
  git worktree add -B gh-pages "$WORK" origin/gh-pages >/dev/null
else
  git worktree add --detach "$WORK" >/dev/null
  git -C "$WORK" checkout --orphan gh-pages >/dev/null
fi

find "$WORK" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -r "$ROOT/site/." "$WORK/"
# Without this, GitHub Pages runs the tree through Jekyll, which drops any file
# or directory whose name begins with an underscore.
touch "$WORK/.nojekyll"

git -C "$WORK" add -A
if git -C "$WORK" diff --cached --quiet; then
  echo "gh-pages is already up to date"
  exit 0
fi
git -C "$WORK" commit -q -m "$MESSAGE"
git -C "$WORK" push -q origin gh-pages
echo "published to gh-pages ($(git -C "$WORK" rev-parse --short HEAD))"
echo "live at https://4waiz.github.io/Dapper/ (Pages can take a minute, and caches for about ten)"
