#include <cassert>

template <unsigned long N>
int f(int x)
{
  auto b = [&](int i) -> int { return i + x; };
  auto a = [](int i) -> int { return i; };
  return b(1) + a(1);
}

int main()
{
  /* 2 is what the capturing lambda returns when it loses its capture. */
  assert(f<2>(5) == 2);
  return 0;
}
