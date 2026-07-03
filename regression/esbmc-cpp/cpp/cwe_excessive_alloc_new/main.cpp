extern unsigned nondet_uint();

int main()
{
  // operator new[n] with attacker-controlled n: the pass scales the element
  // count by sizeof(int), so the byte request is unbounded above the default
  // 1 MiB --excessive-alloc-check limit (CWE-789).
  unsigned n = nondet_uint();
  int *p = new int[n];
  delete[] p;
  return 0;
}
