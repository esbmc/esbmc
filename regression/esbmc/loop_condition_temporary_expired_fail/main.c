struct S { int a[2]; };
int n;
struct S f() { struct S s = {{n < 3, 0}}; n++; return s; }
int main()
{
  int *q;
  while ((q = f().a)[0])
    __ESBMC_assert(q[1] == 0, "temporary read in body");
  return 0;
}
