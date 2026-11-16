struct S { unsigned short a, b, c; };
struct S s[4];
int main(void)
{
  unsigned char *p = (unsigned char *)&s[2] + 4;
  *(unsigned int *)p = 0x00020001u;
  __ESBMC_assert(s[3].a == 0, "next");
  return 0;
}
