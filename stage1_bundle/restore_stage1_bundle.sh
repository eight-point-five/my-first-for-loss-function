#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c SHA256SUMS.txt
cat stage1_full_20260323.tar.zst.part-* | tar --use-compress-program=unzstd -xf - -C ..
