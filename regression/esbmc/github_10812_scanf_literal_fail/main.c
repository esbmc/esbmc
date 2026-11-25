#include <stdio.h>
int main()
{
  char *p = "abc";
  sscanf("x", "%c", p);
  return 0;
}
