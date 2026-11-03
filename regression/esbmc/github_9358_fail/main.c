// #9358: with --printf-check, %s on an array that holds no NUL reads past it.
#include <stdio.h>

int main(void)
{
  char tag[3] = {'a', 'b', 'c'};
  printf("%s\n", tag);
  return 0;
}
