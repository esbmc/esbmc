#include <string.h>

int main()
{
  int a;
  int b;

  char *ptr1 = &a;
  char *ptr2 = &b;
  memcpy(ptr1+1, ptr2+1, sizeof(char));

  assert(a == b);
  return 0;  
}
