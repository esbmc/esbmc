#include <string.h>
#define SIZE 10

int main() {
  const char *source = "Hello";
  char destination[SIZE];
  strncpy(destination, source, SIZE);
  return 0;
}
