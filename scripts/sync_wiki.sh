#!/usr/bin/env bash
# 手动把 docs/wiki/ 同步到 GitHub Wiki。
# 用法：bash scripts/sync_wiki.sh
# 前提：1) Wiki 已在网页创建过第一页；2) 本机已配好 GitHub 推送权限。
set -euo pipefail

WIKI_URL="${WIKI_URL:-https://github.com/Morainet/morainet-ai.wiki.git}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/docs/wiki"

[ -d "$SRC" ] || { echo "找不到 $SRC"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "克隆 wiki: $WIKI_URL"
git clone --quiet "$WIKI_URL" "$TMP"

# 非破坏性：只新增/更新 docs/wiki 里的页面，不删除 wiki 上已有的其它页面。
cp "$SRC"/*.md "$TMP"/

cd "$TMP"
git add -A
if git diff --cached --quiet; then
  echo "无变更，跳过。"
  exit 0
fi
git commit --quiet -m "docs: sync wiki from local docs/wiki"
git push --quiet
echo "✅ 已同步到 GitHub Wiki"
