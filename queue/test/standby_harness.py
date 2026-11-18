#!/usr/bin/env python3
"""`b70 serve --standby` around standby_fake_server.py instead of a model (test/standby.sh): llm/b70's own standby code
(its port gate, queue turns and reloads) against whatever b70q daemon B70Q_SOCKET points at.

usage: standby_harness.py <port> <owner> <fake server log>"""
import argparse
import importlib.machinery
import importlib.util
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
loader = importlib.machinery.SourceFileLoader("b70", os.path.join(here, "..", "..", "llm", "b70"))
spec = importlib.util.spec_from_loader("b70", loader)
assert spec
b70 = importlib.util.module_from_spec(spec)
loader.exec_module(b70)

port, owner, log = int(sys.argv[1]), sys.argv[2], sys.argv[3]
a = argparse.Namespace(port=port, inner_port=b70.free_port(), queue=owner, name="fake-model", model="/tmp/fake-model")
args = [sys.executable, os.path.join(here, "standby_fake_server.py"), str(a.inner_port), log]
warm = {"model": "fake-model", "messages": [{"role": "user", "content": "Hello."}], "max_tokens": 1, "tag": "warm"}
sys.exit(b70.serve_standby(a, args, 1, warm, lambda tid: None))
