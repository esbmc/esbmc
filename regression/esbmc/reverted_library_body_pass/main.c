#include <assert.h>
#include <stdbool.h>

static bool reverted;

bool __ESBMC_reverted(void)
{
  return reverted;
}

int main(void)
{
  reverted = false;
  assert(!__ESBMC_reverted());
  reverted = true;
  assert(__ESBMC_reverted());
  reverted = false;
  assert(!__ESBMC_reverted());
}
