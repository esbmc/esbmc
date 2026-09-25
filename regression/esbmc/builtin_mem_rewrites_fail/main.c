// Negative counterpart of builtin_mem_rewrites: the rewritten builtin has the
// real effect, so a claim contradicting it is refuted rather than vacuously
// held -- which is what the unmodelled, nondet version would have allowed.
struct S
{
  int a;
  int b;
};

int main(void)
{
  struct S e;
  e.a = 5;
  __builtin_memset(&e, 0, sizeof(e));
  __ESBMC_assert(e.a == 5, "must not hold");
  return 0;
}
