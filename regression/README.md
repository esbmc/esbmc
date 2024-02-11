Here is an example to run a given regression suite: 

After configuring the project with cmake (be sure to pass `-DBUILD_TESTING=On -DENABLE_REGRESSION=1`), the tests will be available through [ctest](https://cmake.org/cmake/help/latest/manual/ctest.1.html). Some examples:

- `ctest -j4 -L esbmc-cpp/cpp`. Executes all tests inside esbmc-cpp/cpp with 4 threads.
- `ctest -L esbmc-cpp/*`. Executes all tests matching esbmc-cpp/*.
- `ctest -LE esbmc-cpp*`. Executes all tests except the ones inside esbmc-cpp.
- `ctest --progress`. Show testing progress in one line.

See ctest documentation for the list of available commands.
