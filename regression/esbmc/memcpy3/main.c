#include <string.h>

int main() {
  char *src = "Hello";
  char *dest = NULL;
  memcpy(dest, src, 5);
  return 0;
}

