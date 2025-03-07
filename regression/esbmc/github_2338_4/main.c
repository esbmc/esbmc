#include <assert.h>

int main() {
  unsigned int a = 0xFFFFFFFF;
  unsigned int result = -a;
  assert(result == 1);
  return result;
}

