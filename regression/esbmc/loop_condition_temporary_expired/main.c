struct S { int a[2]; };
int n;
struct S f() { struct S s = {{n < 3, 0}}; n++; return s; }
int main()
{
  int *q;
  int b;
  while ((q = f().a)[0] && (b = q[1], 1))
    __ESBMC_assert(b == 0, "temporary read in its full expression");
  return 0;
}
