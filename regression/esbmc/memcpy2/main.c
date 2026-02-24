#include <string.h>
#include <assert.h>

int main() {
  char *src = "Hello";
  char dest[5];
  char* result = memcpy(dest, src, 5);
  assert(result == dest && dest[4] == 'o');
  return 0;
}

