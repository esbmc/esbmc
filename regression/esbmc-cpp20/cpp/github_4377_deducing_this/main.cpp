#include <cassert>

struct S
{
  int v = 7;
  int g(this S const &self) { return self.v; }
  int add(this S const &self, int x) { return self.v + x; }
};

int main()
{
  S s;
  assert(s.g() == 7);
  assert(s.add(5) == 12);
  return 0;
}
