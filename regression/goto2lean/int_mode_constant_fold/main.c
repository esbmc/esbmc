int main(void)
{
  unsigned char c = 255;
  c = c + 1;
  __ESBMC_assert((int)c == 256, "no wraparound under --ir");
  return 0;
}
