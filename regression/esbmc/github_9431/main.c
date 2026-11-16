#include <assert.h>
#include <stdio.h>
int main(void)
{
  int n = -1, m = -1;
  printf("ab%n", &n);
  assert(n == 2);
  printf("%d%n", 123, &m);
  assert(m == 3);
  char buf[16];
  int k = -1;
  sprintf(buf, "xyz%n!", &k);
  assert(k == 3);
  int r = snprintf(buf, sizeof buf, "%n%d", &n, 45);
  assert(n == 0 && r == 2);
  int x = nondet_int(), w = -1;
  printf("%d%n", x, &w);
  assert(w >= 1 && w <= 11);
  return 0;
}
