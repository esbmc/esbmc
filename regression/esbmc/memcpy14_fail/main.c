#include <string.h>

int main() {
  char src[5] = "1234";
  char dst[2];
  memcpy(dst, src, 5); // overflow
  return 0;
}

