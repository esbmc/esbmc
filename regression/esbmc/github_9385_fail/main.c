// #9385: fgets can return NULL at end-of-file.
#include <assert.h>
#include <stdio.h>

int main(void)
{
  char buf[16];
  char *r = fgets(buf, sizeof buf, stdin);
  assert(r != NULL);
  return 0;
}
