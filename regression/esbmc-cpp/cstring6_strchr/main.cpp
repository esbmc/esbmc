/* strchr example */
#include <iostream>
#include <cstring>

int main ()
{
  char str[] = "This is a sample string";
  char * pch;
  pch=strchr(str,'s');
  while (pch!=0)
  {
    pch=strchr(pch+1,'s');
  }
  return 0;
}



