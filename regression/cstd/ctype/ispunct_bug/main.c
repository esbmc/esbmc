#include <ctype.h>

int main ()
{
  int i=0;
  int cx=0;
  char str[]="Hello welcome";
  while (str[i])
  {
    if (ispunct(str[i])) cx++;
    i++;
  }
  assert(cx == 2);
  return 0;
}
