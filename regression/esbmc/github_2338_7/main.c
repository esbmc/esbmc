#include <assert.h>

int main() {
  unsigned int a = 0x80000001;
  unsigned int result = -a;
  return result;
}
