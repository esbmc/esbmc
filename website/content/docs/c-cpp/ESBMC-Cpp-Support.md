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

# Operational models vs. host headers

By default ESBMC compiles C++ with `-nostdinc++`, so `#include <vector>` resolves
to ESBMC's own operational model (OM) rather than the host standard library. Two
flags change that:

- `--no-abstracted-cpp-includes` — do not include the abstract C++ operational
  models at all.
- `--mix-cpp-host-headers` — keep the host system headers visible *alongside* the
  OMs, so an `#include` not covered by the bundled models falls through to the
  host header. Names defined by both (`char_traits`, `istream`, …) can then
  produce ambiguity errors, so this is opt-in.

# Standard-library coverage (2026-07 update)

The following are covered by passing regression tests under `regression/esbmc-cpp*`.

**Newly modelled headers**

- `<atomic>` ([#6129](https://github.com/esbmc/esbmc/pull/6129))
- `<thread>`, `<mutex>` and `<condition_variable>`, lowered onto ESBMC's pthread
  model so interleaving exploration, deadlock and race detection apply
  ([#6394](https://github.com/esbmc/esbmc/pull/6394)); `std::promise` /
  `std::future` on the same basis ([#6411](https://github.com/esbmc/esbmc/pull/6411))
- `<system_error>` ([#6414](https://github.com/esbmc/esbmc/pull/6414)),
  `<iosfwd>` ([#6417](https://github.com/esbmc/esbmc/pull/6417)), and `<complex>`,
  which is now usable ([#6419](https://github.com/esbmc/esbmc/pull/6419))

**Completed existing headers**

- `<algorithm>` — the C++11 algorithms that were missing
  ([#6435](https://github.com/esbmc/esbmc/pull/6435)); `upper_bound` scanned
  backwards and returned the wrong iterator
  ([#6434](https://github.com/esbmc/esbmc/pull/6434))
- `<numeric>` — `iota`, `gcd`, `lcm`, `reduce`
  ([#6418](https://github.com/esbmc/esbmc/pull/6418))
- `<type_traits>` — the classification traits and `_t` aliases
  ([#6412](https://github.com/esbmc/esbmc/pull/6412))
- `<iterator>` — `iterator_traits` is now accessible and the iterator tags are
  present ([#6420](https://github.com/esbmc/esbmc/pull/6420))
- `<array>` — the required `iterator`/`const_iterator` typedefs
  ([#6416](https://github.com/esbmc/esbmc/pull/6416))
- `std::string_view` — the declared-but-missing search members
  ([#6421](https://github.com/esbmc/esbmc/pull/6421)), plus `string` →
  `string_view` conversion, `hash<string_view>` and the `fstream` string
  overloads ([#6397](https://github.com/esbmc/esbmc/pull/6397))
- `std::valarray` — the declared-but-missing members
  ([#6424](https://github.com/esbmc/esbmc/pull/6424))
- Streams — the standard stream objects (`std::cout` and friends) and
  `ios::widen`/`narrow` ([#6425](https://github.com/esbmc/esbmc/pull/6425)),
  `ios::exceptions` and `ios::copyfmt`
  ([#6429](https://github.com/esbmc/esbmc/pull/6429)); `ostream`'s own
  `width`/`fill`/`precision` were hiding the inherited members and are gone
  ([#6427](https://github.com/esbmc/esbmc/pull/6427))
- `std::variant` and `std::any` — the converting constructor no longer hijacks
  copies ([#6407](https://github.com/esbmc/esbmc/pull/6407),
  [#6408](https://github.com/esbmc/esbmc/pull/6408))
- `std::optional` — `emplace`, `swap`, `std::make_optional`
  ([#6346](https://github.com/esbmc/esbmc/pull/6346))
- `<tuple>` — `std::tie` and `std::ignore`
  ([#6344](https://github.com/esbmc/esbmc/pull/6344)); structured binding over a
  `std::tuple` ([#6342](https://github.com/esbmc/esbmc/pull/6342))

**Containers: const-correctness and the C++11/20 API**

- `std::vector` — the destructor now frees its buffer
  ([#6232](https://github.com/esbmc/esbmc/pull/6232)), plus `data()`,
  `emplace_back`, `shrink_to_fit` and `cbegin`/`cend`
- `std::list` — const `front`/`back`/`rbegin`/`rend`, and `cbegin`/`cend`
- `std::deque` — const iteration and corrected iterator operators
- `std::map`/`std::multimap` — a const `at` overload, `count` taking
  `const key_type&`, const forward and reverse iteration, `emplace`,
  `try_emplace`, `insert_or_assign`, and iterator equality so
  `find(k) != end()` behaves
- `std::set`/`std::multiset` — const-correct const iterators and `emplace`
- C++20 `contains` on `map`/`multimap`/`set`/`multiset`, and `cbegin`/`cend`
  across `deque`/`set`/`multiset`/`map`/`multimap`

**`std::string`**

- The `(const char*, size_t)` range constructor and the fill constructor
  (whose parameters were reversed) are fixed
- `operator<` / `operator>` are length-aware and available as free non-member
  overloads against `const char*`
- `substr(pos, n)` is `const`; C++20 `starts_with` / `ends_with` are available
- `at` throws a catchable `std::out_of_range` instead of asserting
  ([#6463](https://github.com/esbmc/esbmc/pull/6463))

**Standard-version guards.** `<compare>` is includable before C++20, `<bit>`,
`<span>` and `<expected>` are guarded, `<array>` is usable in C++11, `<limits>`
works under `--std c++11` and `c++14`, and the C++98 container headers parse
under `--std c++03`.

# Language and frontend fixes (2026-07 update)

- **Placement new** is modelled, including the form without an initializer
  ([#6133](https://github.com/esbmc/esbmc/pull/6133),
  [#6195](https://github.com/esbmc/esbmc/pull/6195))
- **Virtual dispatch through a non-arrow member expression** — `(*p).f()` and
  `ref.f()` now dispatch virtually
  ([#6272](https://github.com/esbmc/esbmc/pull/6272))
- **`delete`** dispatches through the virtual destructor slot
  ([#6202](https://github.com/esbmc/esbmc/pull/6202)), and deleting through a
  base subobject whose destructor is not virtual is rejected
  ([#6285](https://github.com/esbmc/esbmc/pull/6285))
- **Multiple inheritance** — nested base subobjects, `dynamic_cast<T*>`
  re-offsetting across bases, override thunks keyed by each base's virtual name
  and adjusted by the indirect base's cumulative offset, and catch-by-base
  binding re-offset for MI thrown types
- **`typeid`** on a polymorphic glvalue reports the dynamic type
  ([#6392](https://github.com/esbmc/esbmc/pull/6392)), and the `type_info` name
  string is NUL-terminated
- **Virtual bases** are initialised only in the most-derived constructor
  ([#6159](https://github.com/esbmc/esbmc/pull/6159))
- **Conditional operators** — address-of distributes over a conditional lvalue,
  class-typed conditionals elide their temporaries, and a reference parameter
  binds to the selected arm
- **Dynamic exception specifications** — `unexpected` / `bad_exception` recovery
  ([#6240](https://github.com/esbmc/esbmc/pull/6240)); an aggregate-thrown
  exception gets a symbol-consistent type id
  ([#6303](https://github.com/esbmc/esbmc/pull/6303))

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

