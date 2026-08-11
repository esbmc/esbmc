#include <stdint.h>

struct N
{
  char a;
  uint32_t b;
};

int main(void)
{
  _Alignas(16) struct N s;
  (void)s;
  __ESBMC_assert(((uintptr_t)&s % 32) == 0, "over-claimed 32-byte alignment");
  return 0;
}
