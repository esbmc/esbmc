#include <string.h>

int main ()
{
  char str[] = "This is a sample string";
  char key[] = "aeiou";
  char * pch;

  pch = strpbrk (str, key);
  while (pch != NULL)
  {
    assert((*pch == '2') || (*pch == 'a') || (*pch == 'e'));
    pch = strpbrk (pch+1,key);
  }

  return 0;
}
