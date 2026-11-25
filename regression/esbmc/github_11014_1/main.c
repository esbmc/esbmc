/* #11014: storing a struct over a bit-field struct under --ir */
struct d { unsigned a : 4, b : 4, c : 24; };
struct u { unsigned char b[4]; };
int main(void)
{
  struct d d = {1, 2, 0x332211};
  struct u *p = (struct u *)&d;
  struct u w = {{0x65, 0x44, 0x55, 0x66}};
  *p = w;
  __ESBMC_assert(d.a == 5, "x");
  return 0;
}
