int main(void)
{
  unsigned char c = 256;
  __ESBMC_assert(c == 256, "no wraparound under --ir");
  return 0;
}
