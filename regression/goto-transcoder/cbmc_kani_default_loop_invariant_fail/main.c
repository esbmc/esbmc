/* Cutting the loop at its invariant still reports a violation after it. */
unsigned nondet_uint(void);
int __KANI_marker;
int main(void)
{
  unsigned n = nondet_uint(), i = 0;
  unsigned long long s = 0;
  while (i < n)
    __CPROVER_loop_invariant(i <= n && s == 2ULL * i)
    {
      ++i;
      s += 2;
    }
  __CPROVER_assert(s < 2ULL * n, "s stays below twice n");
  return 0;
}
