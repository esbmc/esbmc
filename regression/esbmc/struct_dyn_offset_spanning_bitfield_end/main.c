struct S { unsigned short x; unsigned short c : 9; unsigned d : 7; };
struct S s[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  *(unsigned short *)((char *)&s[i] + 2) = 0x7FFFu;
  __ESBMC_assert(s[i].c == 0x1FF && s[i].d == 0x3F, "c and d");
  return 0;
}
