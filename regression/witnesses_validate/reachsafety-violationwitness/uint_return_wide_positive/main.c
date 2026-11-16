unsigned int value(void)
{
  return 1;
}

int main(void)
{
  unsigned int x = value();
  __ESBMC_assert(x != 1, "target");
}
