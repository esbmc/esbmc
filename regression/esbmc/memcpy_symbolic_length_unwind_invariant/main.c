/* The cost of a symbolic-length memcpy must not scale with --unwind: no byte
   loop is emitted. Before this was modelled it was 98, 178, 338 and 658 symex
   assignments at --unwind 10, 20, 40 and 80; it is now 29 at every bound; 322 VCCs at --unwind 80 became 4. */
#include <string.h>
#include <assert.h>

int main()
{
  char src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  char dst[8] = {0};

  unsigned long n = nondet_ulong();
  __ESBMC_assume(n <= 8);

  memcpy(dst, src, n);
  assert(n < 1 || dst[0] == 1);
  return 0;
}
