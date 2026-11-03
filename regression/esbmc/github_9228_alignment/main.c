/* github #9228: base addresses keep the alignment of their objects. */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

_Alignas(32) char buf[3];

int main(void)
{
  _Alignas(16) char local[5];
  char *p = malloc(3);
  assert((uintptr_t)buf % 32 == 0);
  assert((uintptr_t)local % 16 == 0);
  assert(p == NULL || (uintptr_t)p % 16 == 0);
  return 0;
}
