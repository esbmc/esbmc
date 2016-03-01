#include <assert.h>
#include <ctype.h>

int main ()
{
  char str[]="s1776ad";

  int i = 0;
  while (i < 4)
    assert(isdigit(str[i++]));

  return 0;
}
