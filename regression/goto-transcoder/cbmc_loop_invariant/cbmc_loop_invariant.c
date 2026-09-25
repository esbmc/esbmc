int main()
{
  int n;
  __CPROVER_assume(n >= 0 && n <= 5);
  int s = 0;
  for (int i = 0; i < n; i++)
    __CPROVER_loop_invariant(i >= 0 && i <= n && s == i)
  {
    s += 1;
  }
  __CPROVER_assert(s == n, "the loop counts to n");
  return 0;
}
