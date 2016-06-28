#!/usr/bin/env bash
echo "Running tests..."
py.test --doctest-modules deep-deep/deepdeep

echo "Running type checks..."
mypy --silent-imports deep-deep/deepdeep/
