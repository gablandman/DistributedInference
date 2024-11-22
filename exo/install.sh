#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate
uv pip install -e .
 