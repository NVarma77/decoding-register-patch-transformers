#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
release_tag=${1:-v0.1.0}
github_repository=${GITHUB_REPOSITORY:-NVarma77/null-problem-sae-ablation}
download_dir=${ARTIFACT_DOWNLOAD_DIR:-"$repo_root/release-assets/downloads/$release_tag"}

for command_name in gh sha256sum tar unzstd; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "required command not found: $command_name" >&2
        exit 1
    fi
done

mkdir -p "$download_dir"

gh release download "$release_tag" \
    --repo "$github_repository" \
    --dir "$download_dir" \
    --pattern '*.tar.zst' \
    --pattern 'SHA256SUMS' \
    --pattern 'ARTIFACT_CONTENTS.txt' \
    --clobber

(
    cd "$download_dir"
    sha256sum -c SHA256SUMS
)

results_archive="$download_dir/null-problem-sae-ablation-results-${release_tag}.tar.zst"
checkpoints_archive="$download_dir/null-problem-sae-ablation-checkpoints-${release_tag}.tar.zst"

tar --extract --file="$results_archive" --directory="$repo_root" --use-compress-program=unzstd
tar --extract --file="$checkpoints_archive" --directory="$repo_root" --use-compress-program=unzstd

echo "verified and restored release $release_tag into $repo_root"
