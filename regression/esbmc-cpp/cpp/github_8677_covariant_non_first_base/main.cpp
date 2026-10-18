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
  Y *py = b->make();
  assert(py == static_cast<Y *>(&gz));
  assert(py->y == 2);
  Base base;
  assert(static_cast<Base *>(&base)->make() == &gy);
  return 0;
}
