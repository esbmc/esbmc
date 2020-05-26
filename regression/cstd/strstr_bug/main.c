#include <string.h>

int main ()
{
  char str[] ="This is a simple string";
  char * pch;
  pch = strstr (str,"simple");
  strncpy (pch,"sample",5);
  assert(!strcmp(str, "This is a sample string"));
  return 0;
}
