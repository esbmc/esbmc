// A recursive function with a reachable base case (n <= 0) that simply
// recurses deeper than the unwind bound. This is a verifier k-bound
// limitation, not a weakness: the comment stays "recursion unwinding
// assertion" and is intentionally left unmapped to any CWE.
int f(int n)
{
  if (n <= 0)
    return 0;
  return f(n - 1);
}

int main(void)
{
  return f(1000);
}
