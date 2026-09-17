---
title: C++ Support
aliases:
  - /docs/c-cpp/esbmc-cpp-support/
---

This page is a reference of the C++ language features and standard-library
headers supported by ESBMC's Clang-based C++ frontend. For what is *not*
supported, see [Limitations](./limitations).

## At a glance

| Standard | Language features | Library | Regression suite |
| --- | --- | --- | --- |
| C++98 / C++03 | Full | Containers, streams, strings | `regression/esbmc-cpp` |
| C++11 | Full | Move semantics, smart pointers, `<chrono>`, `<atomic>`, `<thread>` | `regression/esbmc-cpp11` |
| C++14 | Full | `std::make_unique` | `regression/esbmc-cpp14` |
| C++17 | Full | `std::optional`, `std::variant`, `std::any`, `std::string_view` | `regression/esbmc-cpp17` |
| C++20 | Broad | `std::span`, `<compare>`, `std::source_location` | `regression/esbmc-cpp20` |
| C++23 | Selected | `std::expected` | `regression/esbmc-cpp20` |

The default is C++17. Select a different standard with `--std`, for example
`--std c++20` or `--std c++03`.

Support is not gap-free at any standard — see [Limitations](./limitations) for
the known exceptions, notably constructor and destructor ordering.

## Verifying a C++ program

ESBMC checks memory safety, arithmetic overflow, pointer safety and user
assertions across *all* inputs, rather than the single input a test would
exercise. Given `shape.cpp`:

```cpp
#include <cassert>

class Shape
{
public:
  virtual int area() const = 0;
  virtual ~Shape() = default;
};

class Square : public Shape
{
  int side;

public:
  Square(int s) : side(s) {}
  int area() const override { return side * side; }
};

int main()
{
  int n = nondet_int();
  __ESBMC_assume(n >= 1 && n < 100);

  Shape *s = new Square(n);
  assert(s->area() > n);
  delete s;
  return 0;
}
```

Run it with:

```bash
esbmc shape.cpp --incremental-bmc
```

ESBMC finds the one input that breaks the assertion — `n == 1`, where
`1 * 1` is not greater than `1`:

```
[Counterexample]

State 1 file shape.cpp line 21 column 3 function main thread 0
----------------------------------------------------
  n = 1 (00000000 00000000 00000000 00000001)

State 5 file shape.cpp line 25 column 3 function main thread 0
----------------------------------------------------
Violated property:
  file shape.cpp line 25 column 3 function main
  assertion main
  !((_Bool)((signed long int)(!(return_value$ > n))))

VERIFICATION FAILED
```

Tightening the assumption to `n > 1` makes the property hold for every one of
the remaining inputs, and ESBMC reports `VERIFICATION SUCCESSFUL`.

See the [Usage](/docs/usage) guide for the full set of options and the
[Constructs](/docs/constructs) reference for `__ESBMC_assume`, `nondet_int` and
the other verification annotations.

## Operational models vs. host headers

By default ESBMC compiles C++ with `-nostdinc++`, so `#include <vector>`
resolves to ESBMC's own operational model (OM) rather than your host standard
library. OMs are written to be verification-friendly — small, loop-free where
possible, and free of the inline assembly and compiler intrinsics that real
standard-library implementations rely on.

Two flags change this behaviour:

| Flag | Effect |
| --- | --- |
| `--no-abstracted-cpp-includes` | Do not include the abstract C++ operational models at all. |
| `--mix-cpp-host-headers` | Keep the host system headers visible *alongside* the OMs, so an `#include` not covered by the bundled models falls through to the host header. |

`--mix-cpp-host-headers` is the escape hatch for a header ESBMC does not model
(see [Not modelled](#not-modelled) below). It is opt-in because names defined by
both the OMs and the host library — `char_traits`, `istream` and similar — can
then produce ambiguity errors at parse time.

## Language features

### Classes and objects

- Classes, member functions, member access control
- Constructors and destructors, including default, copy and move forms
- Static and non-static data members; default member initialisers
- Unions, including anonymous unions
- `typeid` on a polymorphic glvalue reports the *dynamic* type, and the
  `type_info` name string is NUL-terminated

### Constructors and destructors

- Delegating and inheriting constructors (`using Base::Base;`), including
  multi-base and chained forms
- Virtual bases are initialised only in the most-derived constructor
- Destructor chains for members and bases, including virtual base destructors
- Destructors run on return paths, for discarded results, and for temporaries
  destroyed mid-expression

{{< callout type="warning" >}}
The order in which constructors and destructors run is not correct in every
case. See [Limitations](./limitations#constructor-and-destructor-ordering).
{{< /callout >}}

### References, temporaries and move semantics

- Lvalue and rvalue references
- Move constructors and move assignment operators
- Address-of distributes over a conditional lvalue; class-typed conditionals
  elide their temporaries; a reference parameter binds to the selected arm
- An lvalue conditional over reference-returning calls stays a reference rather
  than being copied into a temporary, so a write through it — `(c ? get(v) :
  get(w)).x = 2;` — reaches the selected object

### Lambdas

- Capturing and non-capturing lambdas, including generic (`auto`-parameter) forms
- A captureless lambda converted to a function pointer produces a *callable*
  pointer: both the conversion operator and the static invoker behind it get a
  body, so `int (*f)(int) = [](int x){ return x + 1; }; f(2);` runs
- A lambda inside a template instantiation gets its own closure type per
  instantiation, so two instantiations of the same function template do not
  share (and corrupt) one capture layout

### Templates

- Class and function templates, partial and explicit specialisation
- Variable templates (C++14) and variable template declarations (C++17)
- Class template argument deduction (CTAD)
- Concepts and `requires` clauses (C++20)

### Inheritance and polymorphism

- Single-level and multi-level polymorphism
- Pure virtual methods and abstract classes
- Virtual inheritance, including the diamond problem
- Multiple inheritance: nested base subobjects, `dynamic_cast<T*>` re-offsetting
  across bases, override thunks keyed by each base's virtual name and adjusted
  by the indirect base's cumulative offset, and catch-by-base binding re-offset
  for multiple-inheritance thrown types
- Under a virtual base, a derived-to-base conversion onto a non-first base is
  displaced, so the base's methods, constructor and destructor address the
  base subobject rather than the derived object's leading storage
- Virtual dispatch through a non-arrow member expression — `(*p).f()` and
  `ref.f()` dispatch virtually

### Memory management

- `new` / `delete` and `new[]` / `delete[]`
- `new T[n]` runs `T`'s constructor on every element and `delete[]` runs its
  destructor; `new T[n]()` value-initialises them
- A user-replaced `operator new` / `operator delete` — global, class-level,
  sized or array form — is called instead of the built-in allocator, so a pool
  allocator that hands out overlapping storage is caught. The aligned
  (`std::align_val_t`) and user-placement forms are not routed and keep the
  built-in path
- Detection of dangling pointers, double `delete` and mismatched operators
- Placement `new`, including the form without an initialiser
- `delete` dispatches through the virtual destructor slot; deleting through a
  base subobject whose destructor is not virtual is rejected

### Exceptions

`try` / `catch` / `throw` are supported directly. Passing `--lower-exceptions`
additionally lowers exception handling into explicit control flow, which enables:

- `noexcept` and `throw()` exception specifications
- Dynamic exception specifications, with `unexpected` / `bad_exception` recovery
- `std::exception_ptr` with `current_exception`, `rethrow_exception` and
  `make_exception_ptr`, plus nested exceptions (`throw_with_nested`,
  `rethrow_if_nested`)
- `std::uncaught_exception` / `std::uncaught_exceptions`, via a lowered
  per-thread count

Other exception behaviour:

- `dynamic_cast<T&>` throws `std::bad_cast` on a failed reference cast
- `std::vector::at` and `std::string::at` throw `std::out_of_range` rather than
  asserting
- `std::bad_cast` and `std::bad_typeid` derive from `std::exception` with a
  virtual `what()`
- The exception-state globals are thread-local, so each thread raises, catches
  and clears its own in-flight exception independently; concurrently-throwing
  programs are lowered rather than rejected

### Selected C++17 / C++20 / C++23 features

| Feature | Standard |
| --- | --- |
| Structured bindings, including binding by reference and over `std::tuple` | C++17 |
| `if` and `switch` init-statements | C++17 |
| Class template argument deduction for an aggregate | C++20 |
| Three-way comparison `<=>`, including pointer, floating-point and side-effecting operands | C++20 |
| A defaulted friend `operator<=>` / `operator==`, with both operands bound | C++20 |
| Parenthesized aggregate initialization | C++20 |
| `using enum` declarations | C++20 |
| `char8_t` | C++20 |
| Explicit object parameter ("deducing `this`"), including mutating receivers | C++23 |
| `static operator()`, with the implicit object argument skipped | C++23 |

## Standard-library coverage

The headers below are modelled by ESBMC's own operational models and are covered
by regression tests under `regression/esbmc-cpp*`.

### Containers and iterators

| Header | Notes |
| --- | --- |
| `<vector>` | Including `data()`, `emplace_back`, `shrink_to_fit`, `cbegin`/`cend`; the destructor frees its buffer. Elements are constructed into the raw buffer, so a vector of a non-trivial element type works. `reserve()` grows the buffer in place, which keeps `size()` and `capacity()` decidable, so a `push_back` after it costs the same at any `--unwind` |
| `<list>`, `<forward_list>` | Const `front`/`back`/`rbegin`/`rend`, `cbegin`/`cend`; `emplace`, `emplace_back`, `emplace_front`; `reverse_iterator::base()`; the iterator carries its `iterator_traits` typedefs. Both may hold an incomplete element type, so a node that points back at its own container compiles. `list`'s iterator lives at namespace scope, as libc++ spells it, so [basic.lookup.argdep] associates the element type and a user-declared `operator<` over a `list<T>::iterator` is found |
| `<deque>` | Const iteration; lexicographic `<`/`<=`/`>`/`>=` with const-qualified comparators; `back()` returns `reference`, so `d.back() = x` is a write through the container, and `const_iterator` converts from `iterator` |
| `<map>` | Const `at`, `emplace`, `try_emplace`, `insert_or_assign`, C++20 `contains`; const `find`/`count`/`lower_bound`/`upper_bound`/`equal_range`. Const iterators compare by position rather than by a cached pair, so two iterators into equal-keyed entries stay distinct. `mapped_type` may be incomplete |
| `<set>` | Const-correct const iterators, `emplace`, C++20 `contains` |
| `<unordered_map>`, `<unordered_set>` | `<unordered_map>` provides `std::unordered_multimap` and `<unordered_set>` `std::unordered_multiset`. Per [unord.multimap] the multimap's `insert` never rejects an equivalent key, `erase(k)` removes every match and returns how many, and there is no `operator[]`/`at` |
| `<array>` | `iterator` / `const_iterator` typedefs; usable in C++11 |
| `<queue>`, `<stack>`, `<bitset>` | Includes `std::priority_queue` |
| `<iterator>` | `iterator_traits` and the iterator tags; `advance`, `distance`, `next`, `prev` — stepped one element at a time for an iterator that is not random-access, instead of requiring `+=`; the range accessors including the reverse forms `rbegin` / `rend` / `crbegin` / `crend` and the free `size` / `empty` / `data` |
| `<valarray>` | |

`std::multimap` and `std::multiset` track `std::map` and `std::set`, including
`contains` and `cbegin`/`cend`.

`list`, `set`, `multiset`, `map` and `multimap` gained the
[container.requirements] relational operators, comparing lexicographically over
any element type as `vector` and `deque` already did, so `{2} < {1, 3}` is
false. `forward_list` still has none.

The containers take their `Allocator` template parameter — the ordered and
unordered ones, and `list` and `deque` alongside `vector` and `basic_string` —
so a container spelled with an explicit allocator names the same type it does in
a host build, and `get_allocator()` is available. `size_type` is unsigned,
iterator dereference is const-qualified, and `vector`'s iterator-pair
constructor is constrained so it does not hijack `vector<int>(3, 0)`.

### Strings and streams

| Header | Notes |
| --- | --- |
| `<string>` | `(const char*, size_t)` range and fill constructors; length-aware `operator<`/`operator>`/`operator<=`/`operator>=` including free overloads against `const char*`; `const` `substr(pos, n)`; C++20 `starts_with`/`ends_with`; `at` throws `std::out_of_range`; `clear`, `find_last_not_of`; the full `sto*` family (`stoi`, `stol`, `stoll`, `stoul`, `stoull`, `stof`, `stod`, `stold`). The iterators carry their `iterator_traits` typedefs, `compare` takes its argument by const reference, and `operator+` accepts a `const CharT*`. `size`, `resize`, `max_size` and `rfind` agree with the standard, and the constructor and comparison loops run to a concrete trip count so they converge under a bound |
| `<string_view>` | An instantiation of `basic_string_view`, so `wstring_view` and friends name the same template. Search members, `string` → `string_view` conversion, `hash<string_view>` |
| `<iostream>`, `<istream>`, `<ostream>`, `<ios>`, `<iosfwd>` | Standard stream objects, `ios::widen`/`narrow`, `ios::exceptions`, `ios::copyfmt` |
| `<sstream>`, `<fstream>`, `<streambuf>`, `<iomanip>` | `ostringstream` accumulates into the buffer its `str()` reports; the string streams are templated on their character type and `streambuf` is an instantiation of `basic_streambuf`; `operator<<` is modelled for the built-in types; `<iomanip>` has `std::put_time` |
| `<locale>` | |

`char_traits` is reachable without including `<string>`, its single-return
members are `constexpr`, and `char_traits<char>` compares as `unsigned char`
([char.traits.specializations.char] p2).

### Utilities and type support

| Header | Notes |
| --- | --- |
| `<type_traits>` | Classification traits and the `_t` / `_v` forms, including `is_trivial`, `is_standard_layout`, `is_aggregate`, `is_assignable` and the copy/move/destructible variants, `remove_cvref`, `aligned_storage`, `invoke_result`, and the logical traits `conjunction` / `disjunction` / `negation`. Also `is_object`, `is_scalar`, `is_compound`, `is_fundamental`, `rank`, `add_cv`, `add_volatile`, `has_virtual_destructor`, `is_member_object_pointer`, `is_member_function_pointer`, `is_default_constructible`, `is_move_constructible` / `is_move_assignable`, and the `is_nothrow_*` and `is_trivially_*_constructible` families. `is_convertible` is defined by copy-initialization rather than `static_cast`, so an explicit constructor no longer makes it `true` ([meta.rel]). The `_t` aliases that were missing from [meta.trans] are there — `add_volatile_t`, `add_cv_t` and `add_rvalue_reference_t` alongside the traits themselves — and `remove_all_extents` is modelled, with `type_identity` gated on C++20 as P0887R1 specifies |
| `<utility>` | Including `index_sequence_for` and C++23 `std::unreachable` |
| `<functional>`, `<memory>`, `<initializer_list>` | `<functional>` has the transparent operation functors (`plus<>`, `less<>`, …), `std::reference_wrapper` with `ref`/`cref` and its call operator, `std::placeholders`, and a `std::function` whose call target is templated on its signature; `<memory>` has `std::allocate_shared`, a correct default-constructed `unique_ptr`, the `uninitialized_copy` / `uninitialized_fill` family, and an `allocator_traits` that works with a minimal allocator — `rebind_alloc`, the nothrow copy traits, and `construct`/`destroy` templated on the pointee |
| `<tuple>` | `std::tie`, `std::ignore`, structured binding over a tuple, `tuple_size_v` |
| `<optional>` | `emplace`, `swap`, `std::make_optional`; comparison and ordering against a bare value as well as another `optional` |
| `<variant>`, `<any>` | `std::visit` calls the visitor on the currently held alternative; the converting constructor does not hijack copies |
| `<expected>` | C++23 |
| `<compare>` | Includable before C++20 |
| `<source_location>`, `<span>`, `<bit>` | C++20 |
| `<typeinfo>`, `<exception>`, `<stdexcept>`, `<system_error>`, `<new>` | |
| `<limits>` | Works under `--std c++11` and `c++14` |
| `<filesystem>` | `filesystem::u8path`, `path::u8string`, `path::generic_string`; the [fs.path.decompose] members `filename` / `parent_path` / `extension` / `stem`, matching libc++ on the dot-dot, separator-run (`a//b` → `a`, `//b` → `//`) and trailing-period cases; `directory_entry` and `directory_iterator`, which yield a nondeterministic, bounded number of entries synthesised under the base path rather than reading a real filesystem; `std::error_code` is visible through the header |

### Algorithms and numerics

| Header | Notes |
| --- | --- |
| `<algorithm>` | Including the C++11 algorithms and `move_backward` |
| `<numeric>` | `iota`, `gcd`, `lcm`, `reduce` |
| `<cmath>` | The C99 `<cmath>` functions resolve in namespace `std`, as [cmath.syn] requires: the classifiers `std::isnan`, `std::isinf`, `std::isfinite`, `std::isnormal` and `std::signbit` are re-declared as `std::` overloads lowered to ESBMC's native FP intrinsics, and `std::ilogb`, `logb`, `scalbn`, `scalbln`, `fma`, `remquo`, `lround`, `llround`, `lrint`, `llrint`, `nexttoward` and `nan` resolve as well (`ilogb`, `logb` and `nexttoward` have no model in ESBMC's libc and return a nondeterministic value). `fmod`, `remainder` and `remquo` lower to the solver's exact FP remainder rather than being computed as `x - y*(int)(x/y)`, which double-rounded and overflowed the cast for a large quotient |
| `<complex>`, `<random>` | |

### Time

`<chrono>` models `duration` over any `Rep` and `Period`, the `nanoseconds` …
`hours` typedefs, `duration_cast`, and `time_point` with `time_point_cast`.
Mixed-period arithmetic and comparison go through `common_type`, so
`seconds(1) + milliseconds(500)` is `milliseconds(1500)`, and the converting
`duration` constructor is implicit only where the conversion cannot truncate
([time.duration.cons] p2). `duration::zero` / `min` / `max`, `time_point::min` /
`max`, `treat_as_floating_point` and `duration_values` are there as well.
`std::ratio` — reduced per [ratio.ratio] p1, with `ratio_multiply`,
`ratio_divide` and the `nano` / `micro` / `milli` aliases — is declared by
`<chrono>` itself, which also pulls in `<ctime>` for `system_clock::to_time_t`
and `from_time_t`.

`system_clock`, `steady_clock` and `high_resolution_clock` (an alias for
`steady_clock`) share one tick counter that advances by a non-negative
nondeterministic step. A reading is therefore unconstrained rather than a fixed
constant, while `steady_clock` still satisfies [time.clock.steady] — it never
runs backwards between two `now()` calls. `system_clock::period` follows the
target's standard library — nanoseconds on Linux, microseconds on Apple
platforms, 100 ns on Windows — because the point at which the representation
saturates is observable in verification.

### Concurrency

`<thread>`, `<mutex>`, `<shared_mutex>` and `<condition_variable>` are lowered
onto ESBMC's pthread model, so interleaving exploration, deadlock detection and
data-race detection apply to `std::thread` programs. `<shared_mutex>` sits on the
pthread rwlock model, giving `shared_mutex`, `shared_timed_mutex`, `shared_lock`
and the shared/exclusive locking split. `std::this_thread` and
`std::hash<std::thread::id>` are available. `std::promise` and `std::future` are
modelled on the same basis, and `<atomic>` is modelled with atomic sections.

### C library headers

`<cassert>`, `<cctype>`, `<cerrno>`, `<cfloat>`, `<ciso646>`, `<climits>`,
`<clocale>`, `<cmath>`, `<csetjmp>`, `<csignal>`, `<cstdarg>`, `<cstddef>`,
`<cstdint>`, `<cstdio>`, `<cstdlib>`, `<cstring>`, `<ctime>` and `<cwchar>`
are available.

Their names are declared in namespace `std` as the standard requires, not only
in the global namespace: `std::isalpha`, `std::tolower`, `std::time`,
`std::setlocale`, the C99 `<cmath>` functions (`std::fabs`, `std::sqrt`,
`std::fmod`, …), the `strto*` family, and `div` / `ldiv` / `lldiv` all resolve.

### Standard-version guards

`<compare>` is includable before C++20; `<bit>`, `<span>` and `<expected>` are
guarded by standard version; `<array>` is usable in C++11; and `<limits>` works
under every mode from `--std c++03` up.

Under `--std c++03` the headers that postdate C++03 — `<any>`, `<chrono>`,
`<filesystem>`, `<initializer_list>`, `<optional>`, `<random>`,
`<source_location>`, `<string_view>`, `<unordered_map>`, `<unordered_set>` and
`<variant>` — are *inert* rather than a parse error, so one can sit in a
translation unit that only uses the C++03 headers. Using the type it declares
is still an error in that mode.

### Not modelled

The following standard headers have no operational model. Including one either
fails to resolve, or falls through to your host header under
`--mix-cpp-host-headers` — in which case ESBMC has to verify the real
implementation, which is frequently intractable.

`<forward_list>`, `<regex>`, `<ranges>`, `<format>`, `<concepts>`,
`<coroutine>`, `<charconv>`, `<numbers>`, `<ratio>`, `<typeindex>`,
`<barrier>`, `<latch>`, `<semaphore>`, `<stop_token>`,
`<syncstream>`, `<execution>`, `<memory_resource>`, `<scoped_allocator>`,
`<cwctype>`, `<cfenv>`, `<cinttypes>`.

Note that `<concepts>` being unmodelled does not affect the *language* feature —
concepts and `requires` clauses are supported, as listed above. Likewise
`<ratio>` is not includable, but `std::ratio` and its arithmetic aliases are
declared by `<chrono>` — see [Time](#time).

## Current status

Feature support is tracked on the issue tracker under the
[`C++` label](https://github.com/esbmc/esbmc/issues?q=is%3Aissue+is%3Aopen+label%3AC%2B%2B).
For the current pass rate, run the `esbmc-cpp*` regression suites or consult the
[CI results](https://github.com/esbmc/esbmc/actions).

Maintainers: see [C++ Workflow and Resources](./esbmc-cpp-workflow-and-resources)
for benchmark tracking and the development workflow.
