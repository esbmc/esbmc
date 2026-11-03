#include <cassert>
struct X { int x = 1; };
struct Y { int y = 2; };
struct Z : X, Y { };
Z gz;
Y gy;
struct Base { virtual Y *make() { return &gy; } };
struct Der : Base { Z *make() override { return &gz; } };
int main()
{
  Der d;
  Base *b = &d;
  assert(b->make()->y == 1);
  return 0;
}
