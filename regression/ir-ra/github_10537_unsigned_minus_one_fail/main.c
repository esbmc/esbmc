int main()
{
  /* Under --ir, converting -1 to unsigned does not wrap to UINT_MAX. */
  unsigned u = -1;
  __ESBMC_assert(u > 5, "u > 5");
  return 0;
}
