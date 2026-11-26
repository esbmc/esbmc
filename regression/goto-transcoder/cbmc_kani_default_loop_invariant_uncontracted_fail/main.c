/* The contract cuts only its own loop: the next loop keeps its iterations. */
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
  for (unsigned j = 0; j < 10; ++j)
    __CPROVER_assert(j != 7, "deep bug in uncontracted loop");
  return 0;
}
