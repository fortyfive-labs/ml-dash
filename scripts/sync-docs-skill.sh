#!/usr/bin/env bash
# Refresh skills/dash-docs/ from the published docs site.
#
# The skill is generated from the doc pages at build time (see gen-llms.mjs in
# dreamlake-ai/dash-workspace) and published as a zip alongside the site. This
# repo vendors a copy so the Claude Code plugin works offline and versions with
# the SDK — but the docs site is the source of truth, so never hand-edit
# skills/dash-docs/. Run this instead, then commit the result.
#
#   ./scripts/sync-docs-skill.sh          # refresh from production
#   ./scripts/sync-docs-skill.sh --check  # exit 1 if the vendored copy is stale
set -euo pipefail

URL="${DOCS_SKILL_URL:-https://docs.dash.ml/skills/dash-docs.zip}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/skills"
CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

command -v unzip >/dev/null || { echo "unzip not found" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetching $URL"
curl -fsSL "$URL" -o "$TMP/skill.zip"
unzip -qq "$TMP/skill.zip" -d "$TMP/extracted"

if [[ ! -f "$TMP/extracted/dash-docs/SKILL.md" ]]; then
  echo "unexpected archive layout — expected dash-docs/SKILL.md" >&2
  exit 1
fi

if [[ $CHECK -eq 1 ]]; then
  if diff -rq "$TMP/extracted/dash-docs" "$DEST/dash-docs" >/dev/null 2>&1; then
    echo "skills/dash-docs is up to date."
    exit 0
  fi
  echo "skills/dash-docs is stale vs $URL — run ./scripts/sync-docs-skill.sh" >&2
  diff -rq "$TMP/extracted/dash-docs" "$DEST/dash-docs" || true
  exit 1
fi

rm -rf "$DEST/dash-docs"
mv "$TMP/extracted/dash-docs" "$DEST/dash-docs"
echo "updated skills/dash-docs ($(find "$DEST/dash-docs" -type f | wc -l | tr -d ' ') files)"
