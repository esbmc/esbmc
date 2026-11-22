#include <assert.h>
#include <stddef.h>
#include <stdint.h>

char buf[80];

int main(void)
{
  size_t n = (size_t)PTRDIFF_MAX - 7;
  assert(!__CPROVER_r_ok(buf + 16, n));
  return 0;
}
