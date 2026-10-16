#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
  char b[1200];
  char name[11];

  for (int i = 0; i < 10; i++)
    name[i] = nondet_char();
  name[10] = '\0';

  unsigned char key_len = snprintf(b, sizeof(b), "%s=%d", name, 7);
  assert(key_len == strlen(name) + 2);
  return 0;
}
