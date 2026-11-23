#include <assert.h>
#include <stdint.h>
int main(void)
{
  char *p = (char *)(uintptr_t)5;
  char *q = p - 5;
  assert(q == 0);
  return 0;
}
