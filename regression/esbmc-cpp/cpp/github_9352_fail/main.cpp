// #9352: bool(false + 2) is true, so the compound assignment sets s.b.
#include <cassert>

struct S
{
  bool b : 1;
} s;

int main()
{
  s.b = false;
  s.b += 2;
  assert(!s.b);
  return 0;
}
