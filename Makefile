# 医学 RAG 服务 —— 常用操作入口
#
# 每个目标都只是转发到 tasks.py，保证有 make / 没有 make 的环境行为一致。
# 没有 make 时直接用：
#   python tasks.py verify
#   python tasks.py dev

PY ?= python

.DEFAULT_GOAL := help
.PHONY: help setup setup-test data demo-data verify test lint config-check dev docker clean

help:
	$(PY) tasks.py help

setup:
	$(PY) tasks.py setup

setup-test:
	$(PY) tasks.py setup-test

data:
	$(PY) tasks.py data

demo-data:
	$(PY) tasks.py demo-data

verify:
	$(PY) tasks.py verify

test:
	$(PY) tasks.py test

lint:
	$(PY) tasks.py lint

config-check:
	$(PY) tasks.py config-check

dev:
	$(PY) tasks.py dev

docker:
	$(PY) tasks.py docker

clean:
	$(PY) tasks.py clean
