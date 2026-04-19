#include <string.h>

int main()
{
  char a;
  int b;
  memcpy(&a, &b, sizeof(char) + 1);
  assert(a == b);
  return 0;  
}
