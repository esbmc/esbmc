#include <assert.h>

int a = 1;
int b = 0;

int main() {
  goto c;
d:
  __ESBMC_assume(a == 2);
c:
  b = a;
  a = 0;
  assert(b == 0 || b == 3);
  goto d;
}
