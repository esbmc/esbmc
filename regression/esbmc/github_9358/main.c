// #9358: --printf-check accepts %s on a terminated array, at an offset, and
// on an unterminated one when a precision bounds the read.
#include <stdio.h>

int main(void)
{
  char tag[4] = {'a', 'b', 'c', 0};
  char word[5] = {'x', 0, 'a', 'b', 0};
  char raw[3] = {'a', 'b', 'c'};
  printf("%s %s %.3s\n", tag, word + 2, raw);
  return 0;
}
