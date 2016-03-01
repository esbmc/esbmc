#include <string.h>

int main ()
{
  char str[] = "memmove can be very useful......";
  memmove (str+20,str+15,11);
  assert(strcmp(str, "memmove can be very very useful."));
  return 0;
}
