// esbmc/esbmc#7855, reduced: converting a pointer to its address and back does
// not yield the same pointer when the pointer is outside every known object.
// Every store of a pointer into an untyped byte object goes through this cast.
int main(void)
{
  const void *key;
  unsigned long address = (unsigned long)key;
  const void *back = (const void *)address;

  __ESBMC_assert(back == key, "pointer survives the round trip via its address");
  return 0;
}
