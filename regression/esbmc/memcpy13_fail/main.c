#include <string.h>

int main() {
  char *src = NULL;
  char dst[5];
  memcpy(dst, src, 5); // should fail
  return 0;
}

