#include <cassert>

int marker = 1;

// A class-typed array with static storage is constructed after the adjust pass,
// by a consumer that keys on the converter's `#constructor` marker.
struct R
{
  int *p;
  R() : p(&marker)
  {
  }
};

R g[2];

int main()
{
  static R s[2];
  assert(*g[0].p == 1 && *g[1].p == 1);
  assert(*s[0].p == 1 && *s[1].p == 1);
}
