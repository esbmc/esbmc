#include <assert.h>

int main() {
  unsigned int a = 0x80000001;
  unsigned int result = -a;
  assert(result == 0x7FFFFFFF); // Expected modular negation
  return result;
}
