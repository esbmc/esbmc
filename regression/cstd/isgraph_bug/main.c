#include <ctype.h>

int main ()
{
  int i = 0x10;

  while (i < 0x7F) 
    assert(isgraph(i++));

  return 0;
}
