/* #9354: fwrite starting one byte before its buffer. */
#include <stdio.h>

int main(void)
{
  char b[4] = {'a', 'b', 'c', 'd'};
  fwrite(b - 1, 1, 2, stdout);
  return 0;
}
