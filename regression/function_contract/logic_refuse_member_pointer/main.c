struct S { const int *p; };
__ESBMC_logic int first(struct S s) { return s.p[0]; }
int main()
{
  int a[1] = {1};
  struct S s = {a};
  return first(s);
}
