// #8008
bool nondet_bool();
struct S { int v; };
S a{1}, b{1};
int main()
{
  S *sp = &a;
  bool c = nondet_bool();
  int *r = &(c ? *sp : b).v;
  int t = a.v + 1;
  *r = 5;
  int u = a.v + 1;
  __ESBMC_assert(t == u, "a unchanged");
  return 0;
}
