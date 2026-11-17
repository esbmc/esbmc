// #677: a pointer aligned with a bit mask kept the offset of the unmasked one.
#include <stdint.h>

#define _4KB 0x1000
#define _32KB 0x8000
#define _4KB_MASK (~(_4KB - 1))

int main(void)
{
  void *Heap = __ESBMC_alloca(_32KB);
  __ESBMC_assume((uint64_t)Heap < 0xFF);

  uint64_t *TabPhy = (uint64_t *)(((uint64_t)Heap + _4KB) & _4KB_MASK);
  uint64_t *TabPhy_Entry = (uint64_t *)(((uint64_t)TabPhy + _4KB) & _4KB_MASK);

  TabPhy[0] = (uint64_t)TabPhy_Entry;
  __ESBMC_assert(TabPhy == &TabPhy[0], "i-st of TabPhy is TabPhy[0]");
  int i = 0;
  __ESBMC_assert(*TabPhy == TabPhy[i], "value of TabPhy is TabPhy[i] value");
  return 0;
}
