#include <string.h>

int main()
{
  char a[4];
  char b[4];
  memcpy(a, b, 8);
  return a[0];
}
