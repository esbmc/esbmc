struct S { unsigned short a, b, c; };
struct S s[2][3];
int main(void)
{
  unsigned char *p = (unsigned char *)&s[0][2] + 4;
  *(unsigned int *)p = 0x00020001u;
  __ESBMC_assert(s[0][2].c == 1, "c");
  __ESBMC_assert(s[1][0].a == 0, "next row a: expect FAIL");
  return 0;
}
