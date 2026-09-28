int main(void)
{
  unsigned x = 0;
  while (x < 3u)
  {
    ++x;
    __ESBMC_assert(x <= 3u, "bounded counter");
  }
  return 0;
}
