// Soundness check for the non-scalar uninterpreted-function fallback. A pointer
// argument makes the signature non-scalar, so the result is modelled as a fresh
// nondeterministic value (functional congruence is dropped). That value must be
// genuinely unconstrained: a reachable property that depends on it must still be
// refutable, so the violation is found rather than masked. Previously this call
// aborted ESBMC before any verdict (GitHub #5369).
unsigned long __CPROVER_uninterpreted_hasher(const void *key);

int main()
{
  int x;
  const void *p = &x;
  unsigned long h = __CPROVER_uninterpreted_hasher(p);
  __ESBMC_assert(h != 42, "UF result is unconstrained, so h can be 42");
  return 0;
}
