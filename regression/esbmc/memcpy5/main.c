#include <string.h>
#include <assert.h>

int main() {
  int src[2] = {10, 20};
  int dst[2] = {99, 99};
  memcpy(dst, src, 0);
  assert(dst[0] == 99 && dst[1] == 99); // unchanged
  return 0;
}

