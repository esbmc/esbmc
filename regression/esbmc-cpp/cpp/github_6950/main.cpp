#include <cstdint>

struct N
{
  char a;
  uint32_t b;
};

alignas(16) N g;

int main()
{
  alignas(16) N s;
  (void)s;
  __ESBMC_assert(((uintptr_t)&s % 16) == 0, "alignas(16) local is 16-aligned");
  __ESBMC_assert(((uintptr_t)&g % 16) == 0, "alignas(16) global is 16-aligned");
  return 0;
}
