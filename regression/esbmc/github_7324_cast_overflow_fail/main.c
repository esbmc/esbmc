// A value above INT_MAX does not fit in an int. The SMT lowering compared it
// against 2^32-1 instead of the signed type's own maximum, so the cast
// overflow went unreported (#7324).
int main(void)
{
  unsigned int u;
  __ESBMC_assume(u > 2147483647u && u < 3000000000u);
  int i = (int)u;
  return i;
}
