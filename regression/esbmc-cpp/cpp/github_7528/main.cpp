// A closure's name was keyed by file/line/column plus, since #6969, the
// enclosing *function* template's specialisation. A lambda in a class-template
// member (#7528) and a nested lambda (#7529) have no specialisation args on
// their immediate context, so every instantiation but the first reused one
// closure record and got an operator() with no body.
#include <cassert>

template <class T>
struct S
{
  int g(int x)
  {
    auto a = [](int i) -> int { return i; };
    auto b = [&](int i) -> int { return i + x + (int)sizeof(T); };
    return a(1) + b(1);
  }
};

template <unsigned long N>
int f(int x)
{
  auto outer = [&](int i) -> int {
    auto inner = [&](int j) -> int { return j + x + (int)N; };
    return inner(i);
  };
  return outer(1);
}

int main()
{
  S<char> sc;
  S<int> si;
  assert(sc.g(5) == 1 + (1 + 5 + 1));
  assert(si.g(5) == 1 + (1 + 5 + 4));

  assert(f<2>(5) == 1 + 5 + 2);
  assert(f<3>(5) == 1 + 5 + 3);
  return 0;
}
