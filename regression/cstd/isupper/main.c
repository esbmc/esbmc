#include <ctype.h>

int main ()
{
  int i = 0x41;

  while (i < 0x5A) 
    assert(isupper(i++));

  return 0;
}
