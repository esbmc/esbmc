#include <assert.h>

int main() {
  int a[4];
  char *p = (char *)&a[0];
  assert(__ESBMC_r_ok(p, sizeof(int) * 4));
}
