#include <ctype.h>

int main ()
{
  int i = 0x61;

  while (i < 0x7A) 
    assert(islower(i++));

  return 0;
}
