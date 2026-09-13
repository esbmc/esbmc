#include <cassert>

int marker = 1;

// A local array of class type arrives with one constructor call for the whole
// array. Without the per-element fan-out only element 0 is constructed, so the
// remaining elements' p is never set. Two dimensions, so the fan-out's
// recursion is covered too.
struct R
{
  int *p;
  R() : p(&marker)
  {
  }
};

int main()
{
  R a[2][3];
  assert(*a[0][0].p == 1 && *a[1][2].p == 1);
}
