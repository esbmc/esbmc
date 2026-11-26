/* A claim past a too-weak contract ends UNKNOWN instead of searching forever. */
unsigned nondet_uint(void);
int __KANI_marker;
int main(void)
{
  unsigned n = nondet_uint(), i = 0;
  unsigned long long s = 0;
  while (i < n)
    __CPROVER_loop_invariant(i <= n)
    {
      ++i;
      s += 3;
    }
  __CPROVER_assert(s == 2ULL * n, "s is twice n");
  return 0;
}
