#include <assert.h>

int a, b;
const char s1[] = "x", s2[] = "y";

int main()
{
  int *p = &a;
  // Distinct objects never share an address (C11 6.5.9p6), so symex folds
  // these before the solver sees them.
  assert(p != &b);
  assert((void *)&a != (void *)&b);
  assert(&s1[0] != &s2[0]);
}
