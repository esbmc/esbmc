#include <assert.h>
#include <stdio.h>

int main(void)
{
  char b[1200];
  char name[11];

  for (int i = 0; i < 10; i++)
    name[i] = 'A';
  name[5] = nondet_char();
  name[10] = '\0';

  int r = snprintf(b, sizeof(b), "%s", name);
  assert(r == 10);
  return 0;
}
