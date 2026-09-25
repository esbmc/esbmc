#include <assert.h>

void __CPROVER_array_copy(void *, const void *);
void __CPROVER_array_replace(void *, const void *);
int nondet_int(void);

int main(void)
{
  int s[3] = {1, 2, 3};
  int d[3] = {9, 9, 9};
  __CPROVER_array_copy(d, s);
  assert(d[0] == 1 && d[2] == 3);

  // Same extent, so replace and copy coincide.
  int r[3] = {0, 0, 0};
  __CPROVER_array_replace(r, s);
  assert(r[1] == 2);

  // The copy is a value assignment, not an aliasing of the source.
  s[0] = 8;
  assert(d[0] == 1);

  char cs[2];
  cs[0] = (char)nondet_int();
  cs[1] = cs[0];
  char cd[2] = {0, 0};
  __CPROVER_array_copy(cd, cs);
  assert(cd[0] == cd[1]);

  return 0;
}
