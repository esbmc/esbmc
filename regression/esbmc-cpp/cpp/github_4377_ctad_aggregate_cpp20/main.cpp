// esbmc/esbmc#4377 (C++20): aggregate class template argument deduction. The
// audit filed this as a C++17 gap, but aggregate deduction guides are C++20
// (P1816) -- clang rejects the same program at --std c++17. This pins the
// C++20 behaviour, which is the one that must hold.
#include <cassert>

template <typename T>
struct Box
{
  T t;
};

int main()
{
  Box b{42};
  assert(b.t == 42);

  Box c{'x'};
  assert(c.t == 'x');
  return 0;
}
