#include <string.h>

int main ()
{
  char str[] ="tes";
  char * pch;
  pch = strstr (str,"tes");
  strncpy (pch,"tis",3);
  assert(!strcmp(str, "tis"));
  return 0;
}
