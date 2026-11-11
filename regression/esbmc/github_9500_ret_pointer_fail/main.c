#include <assert.h>
int g;

int f(int n, int *q) {
  g = 0;
  if (n <= 0)
    return 5;
  *q = f(n - 1, q);
  assert(g == 0);
  g = 0;
  return 7;
}

int main(void) {
  f(1, &g);
  return 0;
}
