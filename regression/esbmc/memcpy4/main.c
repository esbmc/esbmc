#include <string.h>
#include <assert.h>

int main() {
  char src[] = "abc";
  char dst[4];
  memcpy(dst, src, 4);
  assert(strcmp(dst, "abc") == 0);
  return 0;
}

