extern unsigned nondet_uint();

int main()
{
  // operator new[n] with attacker-controlled n: the Clang frontend pre-scales
  // the element count by sizeof(int) to a byte request (do_cpp_new), unbounded
  // above the default 1 MiB --excessive-alloc-check limit (CWE-789).
  unsigned n = nondet_uint();
  int *p = new int[n];
  delete[] p;
  return 0;
}
