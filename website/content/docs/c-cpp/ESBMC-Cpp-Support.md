---
title: CPP Support
---

# Supported features: 
ESBMC v7 supports the following C++ features:
- Class
- C++ New and Delete: Dangling pointer, Double delete and Mismatched operators
- constructors and destructors
- lvalue reference
- rvalue reference
- Move semantics: Move constructor, Move assignment operator.
- C++ OM fixed: string_view, algorithm, vector, Typeinfo, iterator, string, queue, numeric, set
- Standard lib: <type_traits>
- Improved references and Temporary object
- New Clang node
   * DefaultInit Expr
   * sizeof Expr
   * C++ 11 nullptr_t Expr
   * C++ 17 variable template declarations
- template
   * [x] Umbrella issue: https://github.com/esbmc/esbmc/issues/989 (original estimate 2-3 weeks, logged 2.5 weeks)
     
- inheritance and polymorphism:
   * [x] single-level polymorhism
   * [x] multi-level polymorhism
   * [x] pure virtual method
   * [x] virtual inheritance (the diamond problem)
   * [ ] correct order of ctors/dtors: 
     * tracked by the umbrella issue [issue 940](https://github.com/esbmc/esbmc/issues/940), affected features are listed below: 
       * Virtual destructor 
       * Base initialization for the most-derivied class
       * Order of destruction in case of object composition (part-whole relationship)

- (TODO) Exception handling:
   * [x] try-catch, throw

# Recently added language and library support

The following features have been added since the support summary above and
are each covered by passing regression tests (under `regression/esbmc-cpp*`).
The standard given is the one the test exercises (`--std`).

**C++11**
- Inheriting constructors — `using Base::Base;`, including multi-base and chained forms (`regression/esbmc-cpp11/constructors/UsingConstructor*`)

**C++17**
- Structured bindings — `auto [a, b] = …`, including binding by reference (`regression/esbmc-cpp17/cpp/github_4377_structured_binding*`)

**C++20**
- Concepts and `requires` clauses (`github_4190_concept_combo`)
- Three-way comparison `<=>` and the `<compare>` operational model, including pointer and side-effecting operands (`github_4377_spaceship*`, `github_4377_compare`)
- Class template argument deduction (CTAD) (`github_4377_ctad`)
- Parenthesized aggregate initialization (`github_4377_paren_init*`)
- `using enum` declarations (`regression/esbmc-cpp/bug_fixes/github_4195*`)
- `char8_t` (`github_4377_char8`)
- Library: `std::span` (`github_4190_span`, `github_4248_span_bit`), `std::optional` (`github_4245_optional_*`), `<chrono>` durations (`github_4245_chrono_*`, `github_4264_chrono_max_*`), `std::source_location` (`github_4377_source_location`)

**C++23**
- Explicit object parameter ("deducing `this`"), including mutating receivers (`github_4377_deducing_this*`)
- `static operator()` — static call operator, with the implicit object argument skipped (`github_4377_static_call`)
- Library: `std::expected` (`github_4377_expected`)

**Standard library**
- `<cmath>` floating-point classifiers `std::isnan`, `std::isinf`, `std::isfinite`, `std::isnormal`, and `std::signbit` — all five are `#undef`-ed from the leaking glibc `<math.h>` macros and re-declared as `std::` overloads lowered to ESBMC's native FP intrinsics (previously `std::isinf`/`std::isfinite`/`std::signbit` failed to parse via `std::__builtin_isinf_sign`) (`regression/esbmc-cpp/bug_fixes/cmath_std_classifiers`)

**Exceptions and destructors**
- `noexcept` / `throw()` exception specifications are lowered under `--lower-exceptions` (`regression/esbmc-cpp/try_catch/lower-exceptions_noexcept_*`, `exception_spec_noexcept_*`)
- `dynamic_cast<T&>` throws `std::bad_cast` on a failed reference cast
- Virtual base destructor and member/base destructor-chain synthesis fixes
- `<exception>` operational model: `std::exception_ptr` with `current_exception` / `rethrow_exception` / `make_exception_ptr`, and nested exceptions (`throw_with_nested` / `rethrow_if_nested`). The per-thread handled-exception stack and uncaught counter are instrumented pay-per-use — only when the program touches them (`lower-exceptions_make_exception_ptr`, `lower-exceptions_exception_ptr_rethrow`, `lower-exceptions_nested_rethrow`)
- `std::uncaught_exception` / `std::uncaught_exceptions` via a lowered per-thread count (`lower-exceptions_uncaught_count`)
- `std::vector::at` throws `std::out_of_range` instead of asserting (`try-catch_vector_02_bug`); `std::bad_cast` / `std::bad_typeid` derive from `std::exception` with a virtual `what()`
- Concurrent exceptions: the exception-state globals are thread-local, so each thread raises, catches, and clears its own in-flight exception independently — concurrently-throwing programs are lowered directly rather than rejected (`lower-exceptions_concurrent`). A pthread start routine reached through a computed pointer (or also called directly) cannot get a sound per-function uncaught-escape check and is declined as unsupported; declining is sound — it never validates a buggy program (`lower-exceptions_concurrent_dualuse`)

# Features WIP:
- Fixing our OMs for STL libraries
   * See guidelines: https://github.com/esbmc/esbmc/wiki/Guidelines-for-Fixing-Operational-Models-(OM)-in-ESBMC
   * OM workload estimate and tracking: https://github.com/esbmc/esbmc/wiki/OM-Workload-Estimate-and-Tracking

# Backlog:
Issues are listed from high to low priority in each subsection: 
- Feature issues: 
  * [] [Order of ctor/dtor - umbrella issue] https://github.com/esbmc/esbmc/issues/940
  * [x] ~~[dangling pointer] https://github.com/esbmc/esbmc/issues/894~~ 
  * [x] ~~[zero initialization] https://github.com/esbmc/esbmc/issues/1103~~ 
  * [Corner cases]:
    * [] [vtable setup for class template] https://github.com/esbmc/esbmc/issues/1127
    * [x] ~~[unamed union ctor] https://github.com/esbmc/esbmc/issues/1128~~
    * [] [function as non-type parameter] https://github.com/esbmc/esbmc/issues/1129
  * [x] ~~[formalisation dynamic alloc] https://github.com/esbmc/esbmc/issues/825~~
  * [x] ~~[new] https://github.com/esbmc/esbmc/issues/924~~
  
- OM issues:
  * [OM simplification] https://github.com/esbmc/esbmc/issues/965

# Development Tracking: 
The new clang-based frontend is under development. We are currently working to pass the benchmark `regression/esbmc-cpp/cpp` which contains 370 test cases. Please see benchmark stats in each subsection. 

The stats are generated by applying the following commands in the benchmark logs from Github workflow "Run a Benchmark": 

To generate stats about error signatures: 
```
egrep "Assertion|ERROR" * -rn | egrep -v "//" | cut -d':' -f3- | sort | uniq -c
```

To generate stats about passes: 
```
egrep "VERIFICATION FAILED|VERIFICATION SUCCESSFUL" * -rn | rev | cut -d ':' -f 1 | rev | sort | uniq -c
```

# Summary of Pass Rate in Each Test Suite: 

> **Note:** the figures below are a historical snapshot from June 2024 and are
> retained for reference only. Pass rates have since improved (see *Recently
> added language and library support* above); for the current state, run the
> `esbmc-cpp*` regression suites or consult the CI results.

| Test Suite | Pass Rate | Date | Remarks |
| --- | --- | --- | --- |
| `cpp` | 333/376, 88.56% | 24/06/2024 | Last run: [result link](https://github.com/esbmc/esbmc/actions/runs/9636043814)|
| `inheritance_bringup`  | 14/15, 93.3% | 24/06/2024 | Skipped as the last TC is not passable even by the old esbmc-v2.1 |
| `polymorphism_bringup` | 39/45, 86.7% | 24/06/2024 | 1x TC failure is caused by [issue 938](https://github.com/esbmc/esbmc/issues/938). 2x TC failures are caused by [issue 940](https://github.com/esbmc/esbmc/issues/940). The remaining failures are skipped as they are not passable even by the old esbmc-v2.1 |
|`cbmc` | 119/120, 99% | 24/06/2024 | This suite contains 'Template_XXX' TCs only. Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5519783478/jobs/10065506571#step:6:310) The failed TC is covered by https://github.com/esbmc/esbmc/issues/940|
|`gcc-template-tests` | 26/32, 81.25% | 24/06/2024 | Last run: [result link](https://github.com/esbmc/esbmc/actions/runs/9636043814) - apart from `arg6`, `const1`, `spec26`, `union1` and `vtable1`, the remaining failed TCs are not passable by v2.1|
|`template` | 17/27, 62.96% | 24/06/2024 | Last run: [result link](https://github.com/esbmc/esbmc/actions/runs/9636043814) - the remaining failed TCs are either OM dependent or not passable by v2.1|
| `stream` | 57/65, 86.15% | 24/06/2024 | Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5748222075)|
|`string` | 206/233, 88% | 24/06/2024 | Last run: [result link](https://github.com/XLiZHI/esbmc/actions/runs/7958289776) |
|`algorithm` | 135/168, 80% | 24/06/2024 | Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5748267716)|
|`deque` | 38/43, 88% | 24/06/2024 | Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5748279315)|
|`list` | 47/72, 65% | 24/06/2024 | Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5748287756)|
|`map` | 38/47, 81% | 24/06/2024 | Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5748293876)|
|`multimap` | 41/45, 91% | 24/06/2024 | Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5748309756)|
|`multiset` | 8/43, 18% | 24/06/2024 | Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5748314369)|
|`priority_queue` | 13/15, 87% | 24/06/2024 | Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5748324357)|
|`set` | 30/48, 62% | 24/06/2024 | Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5748331312)|
|`stack` | 12/14, 85% | 24/06/2024 | Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5748338676)|
|`vector` | 133/149, 89% | 24/06/2024 | Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5748346040)|
|`try_catch` | 73/83, 88% | 24/06/2024 | Last run: [result link](https://github.com/kunjsong01/esbmc/actions/runs/5748354688)|


The following lines are used to fix the TCs, and added here for future reference: 
The commands below are for future reference: 
Fix the include path for Linux CIs: 
```bash
egrep "\-I ~/libraries" . -rl | xargs sed -i 's/-I \~\/libraries/-I \/__w\/esbmc\/esbmc\/src\/cpp\/library/g'
```
Add tag for each TC in a test suite: 
```python
from pathlib import Path
for path in Path('./').rglob('test.desc'):
    print(path)
    f = open(path,'r')
    lines = f.readlines()[:-1]
    lines.append("<item_10_mode>KNOWNBUG</item_10_mode>" + "\n")
    lines.append("</test-case>")
    f.close()
    f = open(path,'w')
    f.writelines(lines)
```


# Tracking Error Signatures for esbmc-cpp/cpp test suite:
Since we've added the support for most of the key features, most, if not all, errors are of the type `PARSE ERRORS` in our OMs for STL test suites. There's no need to track parse errors for those test suites. See section `Summary of Pass Rate in Each Test Suite` for the most up-to-date pass rate. 

# Build LLVM from source:
Using a debug build of clang greatly helps debugging the clang-based C++ converter in ESBMC. 
See Rafael's guide in https://github.com/esbmc/esbmc/wiki/Windows-Build#llvm

