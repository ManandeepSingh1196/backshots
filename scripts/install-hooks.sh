#!/usr/bin/env bash
set -euo pipefail
git config core.hooksPath scripts/hooks
chmod +x scripts/hooks/pre-commit
echo "install-hooks: core.hooksPath set to scripts/hooks"
