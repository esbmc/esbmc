Here is an example to run a given regression suite (e.g., cheri-c): 

```
python3 testing_tool.py --regression cheri-c --tool /Users/lucasccordeiro/esbmc/build/src/esbmc/esbmc --mode CORE
```

* `regression` indicates the suite we want to verify (e.g., `cheri-c`).
* `tool` indicates the location of the binary we want to use.
* `mode` indicates which test cases will be executed; possible values are: `CORE`, `KNOWNBUG`, `FUTURE`, and `THOROUGH`
