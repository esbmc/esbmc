#include <assert.h>

/* The first address past the first page is a valid object address. */
char buf[16];

int main(void)
{
  assert((unsigned long)buf != 4096);
  return 0;
}
