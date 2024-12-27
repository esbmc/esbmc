#include <cassert>

struct a
{
  int b[3];
};
int main()
{
  a d{};
  assert(d.b[0] == 0);
  assert(d.b[1] == 0);
  assert(d.b[2] == 0);
  d.b[0] = 1;
  assert(d.b[0] == 1);
  assert(d.b[1] == 0);
  assert(d.b[2] == 0);

  int i = 0;
  a e{d};
  assert(e.b[0] == 1);
  assert(e.b[1] == 0);
  assert(e.b[2] == 0);
  e.b[2] = 20;
  assert(e.b[0] == 1);
  assert(e.b[1] == 0);
  assert(e.b[2] == 20);

  return 0;
}

