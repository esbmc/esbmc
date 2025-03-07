#include <assert.h>

int main() {
  unsigned int a = 0x00000001;
  unsigned int result = -a;
  assert(result == 0xFFFFFFFF); // Expected result due to modular arithmetic
  return result;
}

