#include <assert.h>

int main() {
  unsigned int a = 0x80000000; // Power of two boundary case
  unsigned int result = -a; 
  assert(result == 0x80000001); // Incorrect assertion
  return result;
}
