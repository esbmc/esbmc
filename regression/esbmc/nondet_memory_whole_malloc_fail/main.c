#include <stdlib.h>

extern void __VERIFIER_nondet_memory(void *, size_t);

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
  s->page[100] = 0;
  __VERIFIER_nondet_memory(s, sizeof(struct state));
  /* every byte is havocked, so this can fail */
  __ESBMC_assert(s->page[100] == 0, "page byte havocked");
  return 0;
}
