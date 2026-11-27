/* As github_10939_maybe_uninit, but asserting the high byte is first. */
#include <assert.h>
#include <stddef.h>
struct cannot_drop { unsigned int c; };
struct manually_drop { unsigned char value[4]; };
union maybe_uninit { struct manually_drop value; };
size_t nondet_size(void);
int main(void)
{
  struct cannot_drop x = {0x11223344u};
  size_t off = nondet_size();
  __CPROVER_assume(off == 0);
  union maybe_uninit a =
    *(union maybe_uninit *)((unsigned char *)&x + off);
  assert(a.value.value[0] == 0x11);
  return 0;
}
