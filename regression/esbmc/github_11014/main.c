/* #11014: reading a bit-field struct as another struct under --ir */
struct d { unsigned a : 4, b : 4, c : 24; };
struct u { unsigned char b[4]; };
int main(void)
{
  struct d d = {1, 2, 0x332211};
  struct u *p = (struct u *)&d;
  struct u v = *p;
  __ESBMC_assert(v.b[0] == 0x21, "x");
  return 0;
}
