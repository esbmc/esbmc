// Mutually recursive f <-> g with no base case: genuinely non-terminating,
// but neither function calls *itself* directly. The CWE-674 detector only
// reasons about direct self-recursion, so it deliberately does NOT flag this
// (sound but incomplete): the comment stays "recursion unwinding assertion"
// and no CWE is emitted. Documents the intentional incompleteness.
int g(int n);

int f(int n)
{
  return g(n + 1);
}

int g(int n)
{
  return f(n + 1);
}

int main(void)
{
  return f(0);
}
