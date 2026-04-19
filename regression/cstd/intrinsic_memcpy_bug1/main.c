#include <string.h>

#define T char

int main()
{
  T a, b;
  memcpy(&a, &b, sizeof(T) + 1);
  assert(a == b);
  return 0;  
}
