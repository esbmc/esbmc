#include <assert.h>

void __CPROVER_array_set(void *, int);
void __CPROVER_array_set_char(void *, char);
int nondet_int(void);

int main(void)
{
  int a[4] = {1, 2, 3, 4};
  __CPROVER_array_set(a, 7);
  assert(a[0] == 7 && a[3] == 7);

  double d[3];
  __CPROVER_array_set(d, 0);
  assert(d[2] == 0.0);

  int s[3];
  int v = nondet_int();
  __CPROVER_array_set(s, v);
  assert(s[0] == s[2]);

  return 0;
}
