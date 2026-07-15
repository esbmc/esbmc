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
  assert(std::invoke(add, 5, 7) == 12);
  assert(std::invoke(scale(), 4) == 12);
  // Negative variant: 3 * 4 is 12, not 13.
  assert(std::invoke([](int a, int b) -> int { return a * b; }, 3, 4) == 13);

  return 0;
}
