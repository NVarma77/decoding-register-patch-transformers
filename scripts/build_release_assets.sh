#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_root=${HF_SAE_SOURCE_ROOT:-$(dirname "$repo_root")}
release_tag=${1:-v0.1.0}
output_dir=${RELEASE_OUTPUT_DIR:-"$repo_root/release-assets"}
source_date_epoch=${SOURCE_DATE_EPOCH:-0}

results_name="null-problem-sae-ablation-results-${release_tag}.tar.zst"
checkpoints_name="null-problem-sae-ablation-checkpoints-${release_tag}.tar.zst"
complete_name="null-problem-sae-ablation-complete-${release_tag}.zip"
results_archive="$output_dir/$results_name"
checkpoints_archive="$output_dir/$checkpoints_name"
complete_archive="$output_dir/$complete_name"

for command_name in cp find git sha256sum sort stat tar touch zip zstd; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "required command not found: $command_name" >&2
        exit 1
    fi
done

mkdir -p "$output_dir"

mapfile -d '' result_files < <(
    cd "$repo_root"
    find results -type f \
        \( -name '*.log' \
        -o -name 'per_image_metrics.csv' \
        -o -name 'selection_audit.json' \
        -o -name 'reconstruction_fidelity.csv' \) \
        -print0 | sort -z
)

if [[ ${#result_files[@]} -eq 0 ]]; then
    echo "no raw result files found under $repo_root/results" >&2
    exit 1
fi

printf '%s\0' "${result_files[@]}" \
    | tar \
        --create \
        --file=- \
        --directory="$repo_root" \
        --null \
        --files-from=- \
        --owner=0 \
        --group=0 \
        --numeric-owner \
        --mtime="@$source_date_epoch" \
    | zstd -10 -T0 --force --quiet -o "$results_archive"

checkpoint_files=()
while IFS= read -r -d '' config_path; do
    relative_config=${config_path#"$repo_root/"}
    relative_checkpoint=${relative_config%/config.json}/ae.pt
    source_checkpoint="$source_root/$relative_checkpoint"
    if [[ ! -f "$source_checkpoint" ]]; then
        echo "missing checkpoint: $source_checkpoint" >&2
        exit 1
    fi
    checkpoint_files+=("$relative_checkpoint")
done < <(find "$repo_root/saes" -type f -name config.json -print0 | sort -z)

if [[ ${#checkpoint_files[@]} -eq 0 ]]; then
    echo "no checkpoint configs found under $repo_root/saes" >&2
    exit 1
fi

printf '%s\0' "${checkpoint_files[@]}" \
    | tar \
        --create \
        --file=- \
        --directory="$source_root" \
        --null \
        --files-from=- \
        --owner=0 \
        --group=0 \
        --numeric-owner \
        --mtime="@$source_date_epoch" \
    | zstd -10 -T0 --force --quiet -o "$checkpoints_archive"

if [[ -n $(git -C "$repo_root" status --porcelain --untracked-files=no) ]]; then
    echo "tracked Git files must be clean before building the complete ZIP" >&2
    exit 1
fi

complete_root_name="null-problem-sae-ablation-${release_tag}"
staging_parent=$(mktemp -d "${TMPDIR:-/tmp}/null-problem-release.XXXXXX")
cleanup() {
    if [[ -n ${staging_parent:-} && -d $staging_parent ]]; then
        rm -r -- "$staging_parent"
    fi
}
trap cleanup EXIT
archive_root="$staging_parent/$complete_root_name"
mkdir -p "$archive_root"

git -C "$repo_root" archive --format=tar HEAD | tar -xf - -C "$archive_root"

for relative_path in "${result_files[@]}"; do
    mkdir -p "$archive_root/$(dirname "$relative_path")"
    cp -a "$repo_root/$relative_path" "$archive_root/$relative_path"
done

for relative_path in "${checkpoint_files[@]}"; do
    mkdir -p "$archive_root/$(dirname "$relative_path")"
    cp -a "$source_root/$relative_path" "$archive_root/$relative_path"
done

mkdir -p "$archive_root/provenance"
git -C "$repo_root" rev-parse HEAD >"$archive_root/provenance/GIT_COMMIT.txt"
git -C "$repo_root" remote get-url origin >"$archive_root/provenance/GIT_REMOTE.txt"
git -C "$repo_root" bundle create \
    "$archive_root/provenance/repository.bundle" --all

manifest="$archive_root/ARCHIVE_MANIFEST.tsv"
{
    printf 'sha256\tbytes\tpath\n'
    while IFS= read -r -d '' staged_file; do
        relative_path=${staged_file#"$archive_root/"}
        read -r digest _ < <(sha256sum "$staged_file")
        printf '%s\t%s\t%s\n' \
            "$digest" "$(stat -c '%s' "$staged_file")" "$relative_path"
    done < <(
        find "$archive_root" -type f ! -name ARCHIVE_MANIFEST.tsv -print0 \
            | sort -z
    )
} >"$manifest"

zip_epoch=$source_date_epoch
if (( zip_epoch < 315532800 )); then
    zip_epoch=315532800
fi
find "$archive_root" -exec touch -h -d "@$zip_epoch" {} +

if [[ -e $complete_archive ]]; then
    rm -- "$complete_archive"
fi
(
    cd "$staging_parent"
    find "$complete_root_name" -type f -print \
        | LC_ALL=C sort \
        | zip -X -9 -q "$complete_archive" -@
)

inventory="$output_dir/ARTIFACT_CONTENTS.txt"
{
    printf 'release_tag\t%s\n' "$release_tag"
    printf 'raw_result_file_count\t%s\n' "${#result_files[@]}"
    printf 'checkpoint_file_count\t%s\n' "${#checkpoint_files[@]}"
    printf 'complete_zip\t%s\n' "$complete_name"
    printf 'source_git_commit\t%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    printf '\n[raw results]\n'
    printf '%s\n' "${result_files[@]}"
    printf '\n[checkpoints]\n'
    printf '%s\n' "${checkpoint_files[@]}"
} >"$inventory"

(
    cd "$output_dir"
    sha256sum "$results_name" "$checkpoints_name" "$complete_name" \
        >SHA256SUMS
    sha256sum -c SHA256SUMS
)

du -h "$results_archive" "$checkpoints_archive" "$complete_archive"
echo "wrote release assets to $output_dir"
