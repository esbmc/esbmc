#include <stdio.h>

int main()
{
  const char* c = __FUNCTION__;
  const char* c1 = __func__;
  const char* c2 = __PRETTY_FUNCTION__;

  const char* __c3;

//  printf("%s", c);
  return 0;
}
