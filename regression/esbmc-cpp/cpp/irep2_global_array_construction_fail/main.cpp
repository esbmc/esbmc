#include <cassert>

int marker = 1;

// The negative half of the static-storage construction pair.
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
  assert(*g[1].p == 2);
}
