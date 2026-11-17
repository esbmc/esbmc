// Widening only at loop back edges left recursion unbounded: the analysis
// recursed until it overflowed the stack.
int sum(int n)
{
  if (n == 0)
    return 0;
  return n + sum(n - 1);
}
int main()
{
  __ESBMC_assert(sum(5) == 14, "sum");
}
