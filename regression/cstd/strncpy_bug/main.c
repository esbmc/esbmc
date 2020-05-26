#include <string.h>

int main ()
{
  char str1[]= "To be or not to be";
  char str2[40];
  char str3[40];

  /* copy to sized buffer (overflow safe): */
  strncpy ( str2, str1, sizeof(str2) );

  /* partial copy (only 5 chars): */
  strncpy ( str3, str2, 5 );
  str3[5] = '\0';   /* null character manually added */

  assert(!strcmp(str1, "To be or not to be"));
  assert(!strcmp(str2, "To be or not to be"));
  assert(!strcmp(str2, "To be"));
  return 0;
}
