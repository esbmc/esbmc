#include <string.h>
#include <assert.h>

int main() {
  char *src = "Hello";
  char dest[5];
  int count = nondet_int();
  __ESBMC__assume(count <= 5 && 0 < count);
  char* result = memcpy(dest, src, count);
  assert(result == dest && dest[count - 1] == 'o');
  return 0;
}

