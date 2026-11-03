/* CBMC's __CPROVER_initialize calls the constructors after the static
 * initialisers, in symbol-name order; a --function harness must too (#8237). */
#include <assert.h>

int g = 3;
int h;

__attribute__((constructor)) static void b_init(void)
{
  g = g * 10 + 2;
}

__attribute__((constructor)) static void a_init(void)
{
  g = g * 10 + 1;
}

__attribute__((constructor)) void z_init(void)
{
  h = g;
}

int harness(void)
{
  assert(g == 3);
  return 0;
}

int main(void)
{
  return 0;
}
