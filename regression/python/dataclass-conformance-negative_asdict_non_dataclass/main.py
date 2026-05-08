from dataclasses import asdict

exception_raised = False
try:
    asdict({"a": 1})
except TypeError as exc:
    exception_raised = True

assert exception_raised
