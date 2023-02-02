Here is an example to run a given regression suite: 

```
python3 testing_tool.py --regression floats --tool /Users/lucascordeiro/esbmc/build/src/esbmc/esbmc --mode CORE --library /Users/lucascordeiro/esbmc/src/cpp/library
```

* `regression` indicates the suite we want to verify (e.g., `floats`).
* `tool` indicates the location of the binary we want to use.
* `mode` indicates which test cases will be executed; possible values are: `CORE`, `KNOWNBUG`, `FUTURE`, and `THOROUGH`.
* `library` indicates the `include` folder with the operational models that ESBMC can use to verify the program.

