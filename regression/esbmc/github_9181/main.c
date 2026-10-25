// https://github.com/esbmc/esbmc/issues/9181
#include <assert.h>

void h(void) {}

int main(void)
{
  assert(!__builtin_constant_p(h()));
  return __builtin_constant_p(h());
}
