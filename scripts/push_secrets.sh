#!/usr/bin/env bash
# Copy the Alpaca credentials from .env into this repo's GitHub Actions
# secrets, so the season can run on GitHub's machines.
#
# Reads .env; never prints a value. Run it once:
#
#     ./scripts/push_secrets.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "No .env here. Run \`comp setup-accounts\` first." >&2
  exit 1
fi
if ! command -v gh >/dev/null; then
  echo "The GitHub CLI (gh) is not installed: https://cli.github.com" >&2
  exit 1
fi

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
echo "Setting secrets on ${REPO}"

KEYS=(
  ALPACA_GROUP_A_KEY_ID ALPACA_GROUP_A_SECRET_KEY
  ALPACA_GROUP_B_KEY_ID ALPACA_GROUP_B_SECRET_KEY
  ALPACA_GROUP_C_KEY_ID ALPACA_GROUP_C_SECRET_KEY
  ALPACA_DATA_KEY_ID    ALPACA_DATA_SECRET_KEY
)

missing=0
for key in "${KEYS[@]}"; do
  # Read the value straight from .env rather than sourcing it, so nothing
  # lands in this shell's environment or its history.
  value="$(grep -E "^${key}=" .env | head -1 | cut -d= -f2- || true)"
  if [[ -z "${value}" ]]; then
    echo "  MISSING ${key} (not in .env)"
    missing=1
    continue
  fi
  printf '%s' "${value}" | gh secret set "${key}" --repo "${REPO}" >/dev/null
  echo "  set ${key}"
done

if (( missing )); then
  echo
  echo "Some secrets were missing. Re-run \`comp setup-accounts\` and try again." >&2
  exit 1
fi

echo
echo "Done. Start the season with:"
echo "    gh workflow run Season --repo ${REPO}"
