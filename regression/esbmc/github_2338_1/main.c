#include <assert.h>

int main() {
  unsigned int a = 0;
  unsigned int result = -a; // Should remain 0 (no overflow)
  assert(!result);
  return result;
}

