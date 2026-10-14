#include <cassert>

int marker = 1;

// A class-typed variable-length array (a GNU extension) has no constant extent,
// so the fan-out declines it. The decline is staged: the declaration keeps its
// initialiser, and element 0 is still constructed. Legacy aborts on this input
// ("cannot determine array size for local ctor init"), so the pair pins the
// flag path only, and only element 0 -- the remaining elements are genuinely
// unconstructed today.
struct R
{
  int *p;
  R() : p(&marker)
  {
  }
};

int main(int argc, char **argv)
{
  int n = argc;
  R a[n];
  R b[2][n];
  assert(*a[0].p == 1 && *b[0][0].p == 1);
}
