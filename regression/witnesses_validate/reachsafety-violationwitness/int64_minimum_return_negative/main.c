long long value(void)
{
  return -9223372036854775807LL - 1;
}

int main(void)
{
  long long x = value();
  __ESBMC_assert(x != (-9223372036854775807LL - 1), "target");
}
