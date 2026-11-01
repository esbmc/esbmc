/* A `?:` around a call lowers to a jump: its value cannot be replayed after
   the havoc, so the invariant is declined rather than assumed stale and the
   loop is unwound. */
int g(int k) { return k; }
int nondet_int();
int main()
{
  int n = nondet_int(), i = 0, x = nondet_int();
  __ESBMC_assume(n >= 0 && n < 4);
  __ESBMC_loop_invariant(0 <= i && i <= n && (x > 0 ? g(i) == i : 1));
  for (; i < n; ++i)
    ;
  return 0;
}
