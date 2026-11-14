int nondet_int(void);
_Bool nondet_bool(void);
struct S { long long v; };
struct W { struct S e[4096]; };
int main(void)
{
  struct W w1, w2;
  int j = nondet_int();
  __ESBMC_assume(j >= 0 && j < 4096);
  w2.e[j].v = 1;
  _Bool b = nondet_bool();
  struct W w = b ? w1 : w2;
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 4096);
  __ESBMC_assert(b || i != j || w.e[i].v == 1, "");
  return 0;
}
