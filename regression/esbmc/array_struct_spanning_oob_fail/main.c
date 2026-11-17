struct S { unsigned short a, b, c; };
struct S s[3];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i == 0 || i == 2);
  unsigned char *p = (unsigned char *)&s[i] + 4;
  *(unsigned int *)p = 1u;
  return 0;
}
