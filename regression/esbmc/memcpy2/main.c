#include <string.h>

int main() {
  char *src = "Hello";
  char dest[5];
  memcpy(dest, src, 5);
  return 0;
}

