// A symbolic index selects the target, so both labels must stay reachable:
// the dispatch chain has to branch rather than collapse onto one label
// (issue #4083). github_4083-invalid-target pins the other side, that a
// target matching no label is caught.
int nondet_int();

int main()
{
  void *labels[] = {&&L1, &&L2};
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  int x = 0;
  goto *labels[i];
L1:
  x = 1;
  goto END;
L2:
  x = 2;
  goto END;
END:
  __ESBMC_assert(x == 1 || x == 2, "reaches exactly one of the two labels");
  return 0;
}
