#!/usr/bin/env bash
set -euxo pipefail

(bash scripts/pages-init.sh)
(bash scripts/pages-pub.sh)
(bash scripts/pages-del.sh)
