---
title: Operational Model Fixing Guideline
---

Any random fix to OMs based on clang error messages may make us fall down the rabbit hole, which could make future fixes more difficult and chaotic. 
This guideline aims to help us to: 
1. Get the big picture of OM include dependencies
2. Get the big picture of the number of errors to be fixed in each OM include dependencies to identify the low-hanging fruits from a vast pool of error signatures. 
   * Enable us to estimate the workload to fix a particular OM
3. Avoid random fixes which could lead to spaghetti and overcomplicated code which is harmful to our software's maintainability and extendibility.
   * The general principle is that If there's a simple fix to the OM, then we don't want to patch ESBMC. Because we don't want to patch ESBMC following a red herring error/assertion/test case (TC) failure.

N.B. Make sure the `clang++` binary you'll be using points to the one used by ESBMC binary, rather than the system's `clang++`. Otherwise, you may get inconsistent error signatures. 

The following subsections uses the example TC `regression/esbmc-cpp/stream/istream_get_1`. So, `test_main.cpp` == `regression/esbmc-cpp/stream/istream_get_1/main.cpp`, and `../library/` == `src/cpp/library`.

# Get a list of included library dependencies for a specific TC
```
clang++ -H test_main.cpp -I ../library/ -ferror-limit=1 -v -w
```
`-I ../library/` points to ESBMC's OM library. <br>
`-H` to print patch of included files in a hierarchical format. `-v` for verbose. `-w` to suppress all warnings. <br>
`-ferror-limit=1` just print one and only one error. 
(clang man page: https://clang.llvm.org/docs/ClangCommandLineReference.html#include-path-management)

Manually deleting the error (if there is any) gives the following list: 
```
. ../library/iostream
.. ../library/iomanip
... ../library/definitions.h
.... ../library/cstddef
..... /usr/lib/llvm-10/lib/clang/10.0.0/include/stddef.h
...... /usr/lib/llvm-10/lib/clang/10.0.0/include/__stddef_max_align_t.h
... ../library/ios
.... ../library/streambuf
..... ../library/cstdio
...... ../library/cstddef
....... /usr/lib/llvm-10/lib/clang/10.0.0/include/stddef.h
...... ../library/cstdarg
.. ../library/istream
.. ../library/ostream
. ../library/fstream
.. ../library/string
... ../library/cstring
... ../library/cassert
... ../library/stdexcept
.... ../library/exception
```

# Get all errors for a specific TC
Do NOT simply use the clang invocation cmd line as it will probably show more errors, such as missing declaration for "__ESBMC__assert" or some intrinsics. 

Use the following patch to view all errors in an OM: 
[show_all_clang_errors_patch.txt](https://github.com/esbmc/esbmc/files/11342971/show_all_clang_errors_patch.txt)

```
./esbmc main.cpp -I library/ > error.txt
```
where main.cpp is the sanity check TC for an OM, library points to our OMs as in src/cpp/library. 

This command outputs everything to `error.txt` file which contains all erorrs reported by clang for each dependency OM. 

# Get the number of errors in each dependency OM for a specifc TC: 
```
egrep "/library.*:[0-9]+:[0-9]+: error" error.txt -o | cut -d':' -f1 | sort | uniq -c
```

Output:
```R
      1 /library/cassert
      2 /library/cstring
      3 /library/exception
     14 /library/fstream
      5 /library/istream
      2 /library/ostream
    232 /library/string
```

Then we know how many errors we need to fix in each dependency OM for a specific test case. See next subsection for guidelines to fix them.

# Guideline to fix errors in each OM:
First we need to separate this OM from the include tree using the following TC template
```cpp
#include <your-target-OM>

int main()
{
  return 0;
}

```

Then fix all errors reported by clang, and submit a PR that enables the above test case. 


# What kind of APIs should a OM provide? 
To answer this question, we need to look at what kind of APIS are provided in the corresponding C++ standard library. The following links gives a good overview of the standard libraries. 

- cppreferenc.com: https://en.cppreference.com/w/cpp/header/cstring
- cplusplus.com: https://cplusplus.com/reference/cstring/ 

### OM dev tracking: 

See wiki page https://github.com/esbmc/esbmc/wiki/OM-Workload-Estimate-and-Tracking
