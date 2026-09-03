#include <cassert>

template <unsigned long N>
int non_capturing_first(int x)
{
  auto a = [](int i) -> int { return i; };
  auto b = [&](int i) -> int { return i + x; };
  return a(1) + b(1);
}

template <unsigned long N>
int capturing_first(int x)
{
  auto b = [&](int i) -> int { return i + x; };
  auto a = [](int i) -> int { return i; };
  return b(1) + a(1);
}

int main()
{
  assert(non_capturing_first<2>(5) == 1 + (1 + 5));
  assert(capturing_first<3>(5) == (1 + 5) + 1);
  return 0;
}
