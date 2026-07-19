#!/usr/bin/env bash
# Fetch the evaluation corpus at a pinned commit.
#
# Retrieval metrics are only comparable across runs if the indexed corpus is
# identical, so the revision is pinned rather than tracking upstream main.
set -euo pipefail

REPO_URL="https://github.com/spring-projects/spring-petclinic.git"
REVISION="f182358d02e4a68e52bdbabf55ca7800288511e7"
TARGET=".eval-corpus/spring-petclinic"

if [ ! -d "$TARGET/.git" ]; then
  mkdir -p "$(dirname "$TARGET")"
  git clone --quiet "$REPO_URL" "$TARGET"
fi

git -C "$TARGET" fetch --quiet origin
git -C "$TARGET" checkout --quiet "$REVISION"

echo "Corpus ready at $TARGET (pinned to ${REVISION:0:7})"
echo "Evaluate with:"
echo "  python3 -m scripts.evaluate_retrieval $TARGET/src/main"
