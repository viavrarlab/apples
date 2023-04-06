#!/usr/bin/env bash
set -euxo pipefail

git fetch origin
git fetch hub
mkdir ../apples-temp
cp -r .git ../apples-temp/.git
cd ../apples-temp
git checkout pages
mkdir ../apples-pages
cp -r .git ../apples-pages/.git
cd ../apples-pages
rm -rf ../apples-temp
