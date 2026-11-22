struct S { signed int a : 3; signed int b : 5; unsigned char c; unsigned short d; unsigned int e; };
struct S s;
int main(void)
{
  int k = nondet_int();
  __ESBMC_assume(k == 0 || k == 4);
  *(unsigned int *)((char *)&s + k) = 0x0004F1FFu;
  __ESBMC_assert(k != 0 || (s.a == -1 && s.b == -1 && s.c == 0xF1 && s.d == 4), "all");
  return 0;
}
