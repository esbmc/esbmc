#include <string.h>

#define T char

int foo()
{
  return 42;
}

int main()
{
  T a, b;
  memcpy(&a, &foo, sizeof(T) + 1);
  assert(a == b);
  return 0;  
}
