#include <string.h>
#include <assert.h>

int main() {
  char src[5] = "test";
  char dst[5];
  int n;
  __ESBMC_assume(n >= 0 && n <= 4);
  memcpy(dst, src, n);
  return 0;
}

