union U { unsigned short a[3]; };
union U u[4];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i == 0 || i == 2);
  unsigned char *p = (unsigned char *)&u[i] + 4;
  *(unsigned int *)p = 0x00020001u;
  __ESBMC_assert(u[i + 1].a[0] == 0, "expect FAIL");
  return 0;
}
