#include <string.h>
#include <assert.h>

int main ()
{
  char str[] = "This is a sample string";
  char * pch;

  pch=strchr(str,'s');
  while (pch!=NULL)
  {
    assert(((pch-str+1) == 3));
    pch=strchr(pch+1,'s');
  }
  return 0;
}
