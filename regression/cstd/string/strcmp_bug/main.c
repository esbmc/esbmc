#include <string.h>

int main ()
{
  char key[] = "apple";
  char buffer[10];
  do {
     scanf ("%79s",buffer);
  } while (strcmp (key,buffer) != 0);

  assert(strcmp (key,buffer));

  return 0;
}
