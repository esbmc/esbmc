// A recursive function with no reachable base case: every path to a return
// goes through the recursive self-call, so the recursion is genuinely
// unbounded (CWE-674 Uncontrolled Recursion) rather than merely deep.
int f(int n)
{
  return f(n + 1);
}

int main(void)
{
  return f(0);
}
