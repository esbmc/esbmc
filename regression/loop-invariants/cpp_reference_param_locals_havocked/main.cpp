// The function's USR has a '$' for the reference parameter; its locals must
// still be havocked, or the step only checks the first iteration.
struct S { int n; };
void f(S &s)
{
  int i = 0;
  int k = 0;
  __ESBMC_loop_invariant(0 <= i && i <= s.n && k == 0);
  for (; i < s.n; ++i)
    k = i;
}
int nondet_int();
int main()
{
  S s;
  s.n = nondet_int();
  __ESBMC_assume(s.n >= 0 && s.n < 100);
  f(s);
  return 0;
}
