// Regression for issue #4760: C++20 concept declarations triggered a
// CONVERSION ERROR ("unrecognized / unimplemented clang declaration
// Concept"). ConceptDecls have no runtime form — clang resolves the
// constraint at template instantiation time — so the converter just
// needs to skip the declaration, like it already does for
// CXXDeductionGuide.
//
// Exercises every shape from the audit (#4377) that was blocked
// pre-fix: bare concept, concept built on a concept, concept in a
// requires-clause, concept as type constraint, concept on a class
// template, and concept in if constexpr.
#include <cassert>
#include <type_traits>

template <typename T>
concept Integral = std::is_integral_v<T>;

template <typename T>
concept Signed = Integral<T> && std::is_signed_v<T>;

template <typename T>
requires Integral<T>
T abs_val(T x)
{
  return x < 0 ? -x : x;
}

template <Signed T>
T negate(T x)
{
  return -x;
}

template <Integral T>
struct Wrapper
{
  T value;
};

template <typename T>
T twice(T x)
{
  if constexpr (Integral<T>)
    return x + x;
  else
    return x * T(2);
}

int main()
{
  assert(abs_val(-5) == 5);
  assert(negate(3) == -3);
  Wrapper<int> w{42};
  assert(w.value == 42);
  assert(twice(7) == 14);
  return 0;
}
