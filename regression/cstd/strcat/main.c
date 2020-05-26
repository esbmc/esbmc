#include <string.h>

int main ()
{
  char str[80];
  strcpy (str,"these ");
  strcat (str,"strings ");
  strcat (str,"are ");
  strcat (str,"concatenated.");

  assert(!strcmp("these strings are concatenated.", str));

  return 0;
}
