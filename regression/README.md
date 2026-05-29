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

When any `CHECK_JSON` directive is present the runner executes ESBMC in a fresh temporary directory so parallel tests cannot clobber each other's output files. The test passes only if every regex matches and every `CHECK_JSON` passes.

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
