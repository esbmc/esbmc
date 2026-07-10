#include <cassert>
#include <functional>

int add(int a, int b)
{
  return a + b;
}

struct scale
{
  int operator()(int x) const
  {
    return 3 * x;
  }
};

int main()
{
  // std::invoke is declared with a trailing return type so that <functional>
  // stays parseable under -std=c++11; check it still deduces correctly.
  assert(std::invoke(add, 5, 7) == 12);
  assert(std::invoke(scale(), 4) == 12);
  assert(std::invoke([](int a, int b) -> int { return a * b; }, 3, 4) == 12);

  return 0;
}
