Here is an example to run a given regression suite: 

After configuring the project with cmake in the build directory, please (be sure to pass `-DBUILD_TESTING=On -DENABLE_REGRESSION=1`), the tests will be available through [ctest](https://cmake.org/cmake/help/latest/manual/ctest.1.html). 

By default, `CORE_REGRESSION_ONLY=ON`. This means only `CORE` tests are added to CTest, and `THOROUGH` tests are ignored unless you explicitly disable this option.

To run the full regression selection (including `THOROUGH`), configure with:

- `-DCORE_REGRESSION_ONLY=OFF`

If you use `./scripts/build.sh`, pass:

- `-R OFF`

You can see below some examples that you can from the build directory:

- `ctest -j4 -L esbmc-cpp/cpp`. Executes all tests inside esbmc-cpp/cpp with 4 threads.
- `ctest -L esbmc-cpp/*`. Executes all tests matching esbmc-cpp/*.
- `ctest -LE esbmc-cpp*`. Executes all tests except the ones inside esbmc-cpp.
- `ctest --progress`. Show testing progress in one line.
- `ctest -j4 -L python --progress --timeout 30`. Sets a timeout of 30s.

We also provide a script to validate the Python regression suite. You can run the following command from `ESBMC_Project/esbmc` directory as:

`./scripts/check_python_tests.sh`

See ctest documentation for the list of available commands.
