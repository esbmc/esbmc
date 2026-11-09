struct S { int a, b, c; };
struct S s;
int c;
int nondet_int(void);
int main(void)
{
  c = nondet_int();
  if (c == 1 || c == 2 || c == 3) return 0;
  *(c != 1 && c != 2 && c != 3 ? &c : (c == 1 ? &s.a : (c == 2 ? &s.b : &s.c))) = 1;
  __ESBMC_assert(s.a == 1, "the store went to c");
  return 0;
}
