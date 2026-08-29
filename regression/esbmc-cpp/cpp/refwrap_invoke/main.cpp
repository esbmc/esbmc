#include <functional>
#include <cassert>

struct adder
{
  int base;
  int operator()(int x) const
  {
    return base + x;
  }
};

int main()
{
  // [refwrap.invoke]: a reference_wrapper around a callable is callable.
  adder a{10};
  std::reference_wrapper<adder> rw(a);
  assert(rw(5) == 15);

  auto lam = [](int x, int y) { return x * y; };
  std::reference_wrapper<decltype(lam)> rl(lam);
  assert(rl(3, 4) == 12);

  // Through a const wrapper, as frame_enforcer.cpp uses it.
  const std::reference_wrapper<decltype(lam)> crl(lam);
  assert(crl(2, 8) == 16);

  // get() and the conversion still work.
  assert(rw.get().base == 10);
  return 0;
}
