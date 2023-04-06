#!/usr/bin/env bash
set -euxo pipefail

parcel build --public-url . js/index.html
