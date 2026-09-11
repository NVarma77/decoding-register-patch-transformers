#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_root=${HF_SAE_SOURCE_ROOT:-$(dirname "$repo_root")}
release_tag=${1:-v0.1.0}
output_dir=${RELEASE_OUTPUT_DIR:-"$repo_root/release-assets"}
source_date_epoch=${SOURCE_DATE_EPOCH:-0}

results_name="null-problem-sae-ablation-results-${release_tag}.tar.zst"
checkpoints_name="null-problem-sae-ablation-checkpoints-${release_tag}.tar.zst"
results_archive="$output_dir/$results_name"
checkpoints_archive="$output_dir/$checkpoints_name"

for command_name in find sort tar zstd sha256sum; do
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

inventory="$output_dir/ARTIFACT_CONTENTS.txt"
{
    printf 'release_tag\t%s\n' "$release_tag"
    printf 'raw_result_file_count\t%s\n' "${#result_files[@]}"
    printf 'checkpoint_file_count\t%s\n' "${#checkpoint_files[@]}"
    printf '\n[raw results]\n'
    printf '%s\n' "${result_files[@]}"
    printf '\n[checkpoints]\n'
    printf '%s\n' "${checkpoint_files[@]}"
} >"$inventory"

(
    cd "$output_dir"
    sha256sum "$results_name" "$checkpoints_name" >SHA256SUMS
    sha256sum -c SHA256SUMS
)

du -h "$results_archive" "$checkpoints_archive"
echo "wrote release assets to $output_dir"
