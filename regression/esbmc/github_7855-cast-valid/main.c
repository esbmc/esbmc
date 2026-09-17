// The github_7855-cast round trip holds once the pointer is pinned to an
// object ESBMC knows about. Pairs with github_7855-cast: a fix for that one
// must not be bought by weakening this.
int g;

int main(void)
{
  const void *key;
  __ESBMC_assume(key == &g || key == 0);
  unsigned long address = (unsigned long)key;
  const void *back = (const void *)address;

  __ESBMC_assert(back == key, "pointer survives the round trip via its address");
  return 0;
}
