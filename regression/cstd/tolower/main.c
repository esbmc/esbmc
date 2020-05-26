#include <ctype.h>

int main ()
{
  int i=0;
  char str[]="Test String.\n";
  char str1[]="test string.\n";
  char c, c1;
  while (str[i])
  {
    c=str[i];
    c1 = str1[i];
    assert(tolower(c) == c1);
    i++;
  }
  return 0;
}
