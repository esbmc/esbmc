// #8743: a constant-evaluated bool true is 1, not -1
#include <cassert>
consteval bool t()
{
  return true;
}
int main()
{
  int y = t() + 1;
  assert(y != 2);
  return 0;
}
