#include <stdio.h>
#include <assert.h>
int main()
{
  char *s = "abcde123415";
  int x = printf("%s\n", s);
  // no body for function __stdio_common_vfprintf
#ifndef _WIN32
  assert(x == 12);
  x+=1;
#endif
  
}