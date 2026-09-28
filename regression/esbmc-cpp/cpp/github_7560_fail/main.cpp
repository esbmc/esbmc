// The initializer must reach the member inside the anonymous struct, so x is
// 3, not 4 (#7560).
#include <cassert>

struct S
{
  struct
  {
    int x;
    int y;
  };
  S() : x(3), y(4)
  {
  }
};

int main()
{
  S s;
  assert(s.x == 4);
  return 0;
}
