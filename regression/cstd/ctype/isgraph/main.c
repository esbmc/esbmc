#include <ctype.h>

int main ()
{
  int i = 0x21;

  while (i < 0x7E) 
    assert(isgraph(i++));

  return 0;
}
