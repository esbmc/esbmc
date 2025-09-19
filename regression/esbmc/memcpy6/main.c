#include <string.h>

int main() {
  char buf[6] = "hello";
  memcpy(buf + 1, buf, 5); // overlap is undefined
  return 0;
}
