#include <ctype.h>

int main ()
{
  char c;
  int i=0;
  char str[]="\t  \t \t    ";
  while (str[i])
  {
    c=str[i];
    assert(isblank(c));
    i++;
  }
  return 0;
}
