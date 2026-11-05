#include <stdlib.h>

extern void __VERIFIER_nondet_memory(void *, size_t);

/* A heap object havocked whole, as the Intel TDX harnesses do. One fresh
 * value covers it; a byte loop over 64 MiB would not finish. */
struct state
{
  unsigned char page[1u << 26];
  unsigned count;
};

int main()
{
  struct state *s = malloc(sizeof(struct state));
  if (!s)
    exit(-1);
  __VERIFIER_nondet_memory(s, sizeof(struct state));
  s->count = 7;
  __ESBMC_assert(s->count == 7, "count kept");
  return 0;
}
