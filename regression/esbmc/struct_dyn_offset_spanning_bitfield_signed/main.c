struct S { signed int a : 3; signed int b : 5; unsigned char c; unsigned short d; };
struct S s[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  *(unsigned int *)&s[i] = 0x0004F1FFu;   /* byte0=0xFF: a=-1, b=-1; c=0xF1; d=4 */
  __ESBMC_assert(s[i].a == -1 && s[i].b == -1 && s[i].c == 0xF1 && s[i].d == 4, "all");
  return 0;
}
