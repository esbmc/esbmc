#include <string.h>

int main ()
{
  char * pch;
  char str[] = "Example string";
  pch = (char*) memchr (str, 'p', strlen(str));

  assert((pch - str + 1) == 4);
  return 0;
}
