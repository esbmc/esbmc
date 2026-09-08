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

template <unsigned long N>
int per_instantiation(int x)
{
  auto b = [&](int i) -> int { return i + x + (int)N; };
  auto a = [](int i) -> int { return i; };
  return b(1) + a(1);
}

/* Captureless siblings collide through their __invoke and conversion-operator
 * USRs too, not just operator(). */
template <unsigned long N>
int via_function_pointer(int x)
{
  int (*p)(int) = [](int i) -> int { return i + 1; };
  int (*q)(int) = [](int i) -> int { return i * 2; };
  return p(x) + q(x);
}

int main()
{
  assert(non_capturing_first<2>(5) == 1 + (1 + 5));
  assert(capturing_first<3>(5) == (1 + 5) + 1);

  assert(per_instantiation<2>(5) == (1 + 5 + 2) + 1);
  assert(per_instantiation<3>(5) == (1 + 5 + 3) + 1);

  assert(via_function_pointer<2>(5) == (5 + 1) + (5 * 2));
  return 0;
}
