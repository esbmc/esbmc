#include <assert.h>
#include <stdio.h>
int main(void)
{
  char name[11];
  for (int i = 0; i < 10; i++)
    name[i] = 'A';
  name[10] = '\0';
  int r = printf("%s", name);
  assert(r == 9);
  return 0;
}
