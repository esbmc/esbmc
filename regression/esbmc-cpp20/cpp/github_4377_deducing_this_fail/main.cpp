#include <cassert>

struct S
{
  int v = 7;
  int g(this S const &self) { return self.v; }
};

int main()
{
  S s;
  // s.g() returns 7; the assertion below must fail.
  assert(s.g() == 8);
  return 0;
}
