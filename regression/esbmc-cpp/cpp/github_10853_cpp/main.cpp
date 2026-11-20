struct S { int a; };
S s;
int main()
{
  unsigned i = 0;
  bool b = true;
  __ESBMC_loop_invariant(i <= 3 && b && s.a == 0);
  while (i < 3)
    i++;
  return 0;
}
