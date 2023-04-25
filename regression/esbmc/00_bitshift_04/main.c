#include <assert.h>

int main() {
  int n = 256 << 1;
  assert(n == 512);

  unsigned char a = 200;
  unsigned char b = a << 4;
  assert(b == 128);

  return 0;
}
