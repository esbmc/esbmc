// github.com/esbmc/esbmc/issues/7643: std::hash<std::thread::id> was reached
// through the non-defining declaration <thread>'s `friend struct hash<id>;`
// creates, and registered as an incomplete struct nothing ever completed.
// All three ingredients are load-bearing: drop the constexpr make_tuple of a
// pointer-to-member below, or make `m` a non-pointer, and clang hands the
// frontend the defining decl instead and the reproducer stops reproducing.
#include <cassert>
#include <thread>
#include <tuple>

struct S
{
  void f()
  {
    std::hash<std::thread::id>{};
  }
  int *m;
};

constexpr auto fields = std::make_tuple(&S::m);

int main()
{
  int x = 1;
  S s;
  s.m = &x;
  s.f();
  assert(*s.m == 2);
  return 0;
}
