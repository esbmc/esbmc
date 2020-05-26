#include <string.h>

int main ()
{
  char str[] = "This is a sample string";
  char * pch;
  pch=strrchr(str,'s');
  assert((pch-str+1) == 11);
  return 0;
}
