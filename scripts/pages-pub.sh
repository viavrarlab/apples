#!/usr/bin/env bash
set -euxo pipefail

rm -rf dist
(bash scripts/parcel-build.sh)
mv dist/* ../apples-pages
cd ../apples-pages
git add .
git commit -m 'Updated pages.'
git push origin pages
git push hub pages
git push via pages
