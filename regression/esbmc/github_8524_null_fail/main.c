#include <assert.h>
#include <stdio.h>

int main(void)
{
  char b[100];
  char name[11] = "AAAAAAAAAA";
  char *p = nondet_int() ? name : 0;

  int r = snprintf(b, sizeof(b), "%s", p);
  assert(r == 10);
  return 0;
}
