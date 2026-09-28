// GCC computed goto: `&&L` takes a label's address and `goto *p` jumps to it.
// The dispatch must land on the label the pointer names -- here labels[1],
// i.e. L2 (issue #4083).
int main()
{
  void *labels[] = {&&L1, &&L2};
  int x = 0;
  goto *labels[1];
L1:
  x = 1;
  goto END;
L2:
  x = 2;
  goto END;
END:
  __ESBMC_assert(x == 2, "computed goto lands on L2");
  return 0;
}
