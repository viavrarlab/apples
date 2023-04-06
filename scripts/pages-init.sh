#!/usr/bin/env bash
set -euxo pipefail

mkdir ../apples-pages
cp -r .git ../apples-pages/.git
cd ../apples-pages
git checkout pages
