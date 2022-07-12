#include <cassert>

int main()
{
  int g;
  int &r = g;
  r = 1;
  assert(r == 1); // pass
  assert(g == 0); // fail
}
