struct S { unsigned int a : 4; unsigned int b : 28; unsigned int c; };
struct S s[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  s[i].c = 9;
  *(unsigned int *)&s[i] = 0x31u;
  /* b == 0 is the unmodelled big-endian store (#10087); 0x31 is the C layout */
  __ESBMC_assert(s[i].a == 0 && (s[i].b == 0 || s[i].b == 0x31), "big-endian layout");
  __ESBMC_assert(s[i].c == 9, "c untouched");
  return 0;
}
