#include <string.h>

int main ()
{
  char szInput[256];
  strcpy(szInput, "just testing");
  assert(strlen(szInput) == 12);
  return 0;
}
