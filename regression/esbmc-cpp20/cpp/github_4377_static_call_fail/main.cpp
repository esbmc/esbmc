#include <cassert>

struct F
{
  static int operator()(int x) { return x + 1; }
};

int main()
{
  F f;
  // f(2) returns 3; the assertion below must fail.
  assert(f(2) == 4);
  return 0;
}
