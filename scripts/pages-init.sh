#!/usr/bin/env bash
set -euxo pipefail

git push origin master
git push hub master
git push via master
mkdir ../apples-temp
cp -r .git ../apples-temp/.git
cd ../apples-temp
git checkout pages
git pull origin pages
git pull hub pages
git pull via pages
mkdir ../apples-pages
cp -r .git ../apples-pages/.git
cd ../apples-pages
rm -rf ../apples-temp
