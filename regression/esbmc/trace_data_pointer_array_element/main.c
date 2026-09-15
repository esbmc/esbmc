// The counterexample printed a pointer read from an array at a symbolic index
// as `&0` rather than the object it points to.
int nondet_int(void);

int a = 1, b = 2, c = 3;
int *ptrs[3];

int main(void)
{
  ptrs[0] = &a;
  ptrs[1] = &b;
  ptrs[2] = &c;
  int idx = nondet_int();
  __ESBMC_assume(idx >= 0 && idx < 3);
  int *p = ptrs[idx];
  __ESBMC_assert(*p != 3, "pointer read at a symbolic index");
  return 0;
}
