int main()
{
  union {
    unsigned char c;
    unsigned int i;
  } u = {0};
  u.i = 0x1200;
  u.c = 5;
  __ESBMC_assert(u.i == 0x1205, "the write to c keeps the upper bytes of i");
  return 0;
}
