#include <ctype.h>

int main ()
{
  int i = 0x31;

  while (i < 0x5A) 
    assert(isupper(i++));

  return 0;
}
