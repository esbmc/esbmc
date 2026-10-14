#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
  char b[1200];
  char name[11];

  for (int i = 0; i < 10; i++)
    name[i] = 'A';
  name[10] = '\0';

  int r = snprintf(b, sizeof(b), "%s", name);
  assert(r == (int)strlen(name));
  assert(sprintf(b, "<%s>", name + 3) == 9);
  return 0;
}
