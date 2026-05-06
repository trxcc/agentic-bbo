#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/package_dbtune_images.sh [options]

Build the two dbtune evaluator images and export them as Docker image tarballs.
The script does not push to Docker Hub.

Options:
  --namespace NAME   Docker Hub namespace/user for the exported tags
                     (default: fakerstrawberry, or DOCKERHUB_NAMESPACE)
  --tag TAG          Image tag (default: v1, or DBTUNE_IMAGE_TAG)
  --out-dir DIR      Output directory (default: dist/docker-images)
  --allow-missing-surrogate-assets
                     Build the surrogate image even if one or more full .joblib
                     checkpoints are missing from bbo/tasks/dbtune/assets/.
  -h, --help         Show this help.

Environment overrides:
  DBTUNE_MARIADB_IMAGE       Full MariaDB image ref.
  DBTUNE_SURROGATE_IMAGE     Full surrogate image ref.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
namespace="${DOCKERHUB_NAMESPACE:-fakerstrawberry}"
tag="${DBTUNE_IMAGE_TAG:-v1}"
out_dir="${DBTUNE_IMAGE_OUT_DIR:-${repo_root}/dist/docker-images}"
allow_missing_surrogate_assets=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace)
      namespace="$2"
      shift 2
      ;;
    --tag)
      tag="$2"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --allow-missing-surrogate-assets)
      allow_missing_surrogate_assets=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not available in PATH" >&2
  exit 1
fi

mariadb_image="${DBTUNE_MARIADB_IMAGE:-${namespace}/agentbbo-dbtune-mariadb-eval:${tag}}"
surrogate_image="${DBTUNE_SURROGATE_IMAGE:-${namespace}/agentbbo-dbtune-surrogate-http-py37:${tag}}"
mariadb_legacy_image="agentbbo-http-mariadb-eval:${tag}"
surrogate_legacy_image="agentbbo-surrogate-http-py37:${tag}"

assets_dir="${repo_root}/bbo/tasks/dbtune/assets"
required_joblibs=(
  RF_SYSBENCH_5knob.joblib
  SYSBENCH_all.joblib
  RF_JOB_5knob.joblib
  JOB_all.joblib
  pg_5.joblib
  pg_20.joblib
)
missing=()
for filename in "${required_joblibs[@]}"; do
  if [[ ! -s "${assets_dir}/${filename}" ]]; then
    missing+=("${filename}")
  fi
done

if [[ "${#missing[@]}" -gt 0 && "${allow_missing_surrogate_assets}" -ne 1 ]]; then
  {
    echo "missing surrogate checkpoint(s) under ${assets_dir}:"
    printf '  - %s\n' "${missing[@]}"
    echo
    echo "Download the full .joblib assets first, or rerun with"
    echo "--allow-missing-surrogate-assets for a smoke-test-only image."
  } >&2
  exit 1
fi

mkdir -p "${out_dir}"
tag_slug="${tag//[^A-Za-z0-9_.-]/_}"
mariadb_tar="${out_dir}/agentbbo-dbtune-mariadb-eval_${tag_slug}.tar"
surrogate_tar="${out_dir}/agentbbo-dbtune-surrogate-http-py37_${tag_slug}.tar"

echo "Building ${mariadb_image} ..."
docker build \
  -t "${mariadb_image}" \
  -t "${mariadb_legacy_image}" \
  "${repo_root}/bbo/tasks/dbtune/docker_mariadb"

echo "Building ${surrogate_image} ..."
docker build \
  -f "${repo_root}/bbo/tasks/dbtune/docker_surrogate/Dockerfile" \
  -t "${surrogate_image}" \
  -t "${surrogate_legacy_image}" \
  "${repo_root}/bbo/tasks/dbtune"

echo "Saving ${mariadb_tar} ..."
docker save -o "${mariadb_tar}" "${mariadb_image}" "${mariadb_legacy_image}"

echo "Saving ${surrogate_tar} ..."
docker save -o "${surrogate_tar}" "${surrogate_image}" "${surrogate_legacy_image}"

if command -v sha256sum >/dev/null 2>&1; then
  (
    cd "${out_dir}"
    sha256sum "$(basename "${mariadb_tar}")" "$(basename "${surrogate_tar}")" > SHA256SUMS
  )
fi

cat <<EOF

Created Docker image packages:
  ${mariadb_tar}
  ${surrogate_tar}

On the upload machine:
  docker load -i ${mariadb_tar}
  docker load -i ${surrogate_tar}
  docker push ${mariadb_image}
  docker push ${surrogate_image}

Compose defaults now use:
  AGENTBBO_DBTUNE_MARIADB_IMAGE=${mariadb_image}
  AGENTBBO_DBTUNE_SURROGATE_IMAGE=${surrogate_image}
EOF
