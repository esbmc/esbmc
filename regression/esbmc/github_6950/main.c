#include <stdint.h>

struct N
{
  char a;
  uint32_t b;
};

struct __attribute__((packed)) P
{
  char a;
  char pad[3];
  uint32_t b;
};

_Alignas(16) struct N g;

int main(void)
{
  _Alignas(16) struct N s;
  __attribute__((aligned(32))) struct N t;
  _Alignas(uint32_t) struct P p;
  uint32_t *q = (uint32_t *)&p.b;
  uint32_t z = *q;
  (void)s;
  (void)t;
  (void)z;

  __ESBMC_assert(((uintptr_t)&s % 16) == 0, "_Alignas(16) local is 16-aligned");
  __ESBMC_assert(((uintptr_t)&g % 16) == 0, "_Alignas(16) global is 16-aligned");
  __ESBMC_assert(((uintptr_t)&t % 32) == 0, "aligned(32) local is 32-aligned");
  __ESBMC_assert(((uintptr_t)q % 4) == 0, "member of _Alignas'd packed struct is 4-aligned");
  return 0;
}
