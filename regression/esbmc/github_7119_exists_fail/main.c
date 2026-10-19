/* github #7119: an __ESBMC_exists assertion prints as written. */
#include <assert.h>

int i;

int main(void)
{
  assert(__ESBMC_exists(&i, i > 0 && i < 0));
  return 0;
}
