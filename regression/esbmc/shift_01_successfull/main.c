#include <assert.h>

int main() {
  int n = 256 << 1;
  assert(n == 512);
  return 0;
}
