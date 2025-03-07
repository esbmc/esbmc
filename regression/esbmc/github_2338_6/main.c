#include <assert.h>

int main() {
  unsigned int a = 0x40000000;
  unsigned int result = -a;
  assert(result == 0xC0000000); // Expected modular negation
  return result;
}
