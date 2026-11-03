#include <string.h>

int nondet_int();

int main()
{
  char a[4] = {0}, b[4] = {0};
  char *p = a;
  if (nondet_int())
    p = 0;
  return memcmp(p, b, 4);
}
