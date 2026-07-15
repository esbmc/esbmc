Here is an example to run a given regression suite: 

After configuring the project with cmake in the build directory, please (be sure to pass `-DBUILD_TESTING=On -DENABLE_REGRESSION=1`), the tests will be available through [ctest](https://cmake.org/cmake/help/latest/manual/ctest.1.html). 

You can see below some examples that you can from the build directory:

- `ctest -j4 -L esbmc-cpp/cpp`. Executes all tests inside esbmc-cpp/cpp with 4 threads.
- `ctest -L esbmc-cpp/*`. Executes all tests matching esbmc-cpp/*.
- `ctest -LE esbmc-cpp*`. Executes all tests except the ones inside esbmc-cpp.
- `ctest --progress`. Show testing progress in one line.
- `ctest -j4 -L python --progress --timeout 30`. Sets a timeout of 30s.

We also provide a script to validate the Python regression suite. You can run the following command from `ESBMC_Project/esbmc` directory as:

`./scripts/check_python_tests.sh`

See ctest documentation for the list of available commands.

A `test.desc` file may also contain `CHECK_JSON` lines, after the stdout/stderr regexes, that assert on the contents of JSON files ESBMC emits (typically `cov-report.json` from `--cov-report-json`). Stdout regexes alone cannot prove the file's contents agree with what the terminal reported. Each line has the form `CHECK_JSON <file> <jsonpath> <op> <literal>`, where:

- `<file>`. Path of the JSON file relative to ESBMC's working directory.
- `<jsonpath>`. Restricted subset of JSONPath using `$`, `.field`, `[index]`, for example `$.summary.covered` or `$.claims[0].status`. No filter expressions are supported.
- `<op>`. One of `==`, `!=`, `<`, `>`, `<=`, `>=`.
- `<literal>`. A JSON literal: number, `"string"`, `true`, `false`, or `null`.

A `test.desc` file may also contain `CHECK_FILE` lines for non-JSON output files ESBMC writes (for example an SMT-LIB2 dump from `--output <file>`), which stdout/stderr regexes cannot reach. Each line has the form `CHECK_FILE <file> <op> <regex>`, where:

- `<file>`. Path of the file relative to ESBMC's working directory.
- `<op>`. Either `contains` (the regex must match somewhere in the file) or `absent` (the regex must not match).
- `<regex>`. A Python regex matched with `re.MULTILINE`; the rest of the line, so it may contain spaces.

When any `CHECK_JSON` or `CHECK_FILE` directive is present the runner executes ESBMC in a fresh temporary directory so parallel tests cannot clobber each other's output files. The test passes only if every regex matches and every `CHECK_JSON`/`CHECK_FILE` passes.

For example, `regression/smtlib/github_6059/test.desc` asserts the `--output` dump holds the real formula and not the status string that used to overwrite it:

```
CORE
main.c
--smtlib --smt-formula-only --output out.smt2
^SMT formula written to output file out\.smt2$
CHECK_FILE out.smt2 contains \(check-sat\)
CHECK_FILE out.smt2 contains \(assert
CHECK_FILE out.smt2 absent SMT formula dumped successfully
```

For example, `regression/goto-coverage/k_path_cov_json_1/test.desc` is:

```
CORE
main.c
--k-path-coverage=3 --cov-report-json
^k-Path Coverage: 12\.5%$
^Coverage report written to cov-report\.json$
^VERIFICATION FAILED$
CHECK_JSON cov-report.json $.coverage_type == "k-path"
CHECK_JSON cov-report.json $.summary.total == 8
CHECK_JSON cov-report.json $.summary.covered == 1
CHECK_JSON cov-report.json $.summary.percentage == 12.5
CHECK_JSON cov-report.json $.claims[13].status == "covered"
```
