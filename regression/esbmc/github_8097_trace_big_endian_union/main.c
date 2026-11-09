/* #8097: on a big-endian target a union's members share its lowest-addressed
 * bytes, so .c is the most significant byte of .i. */
unsigned int nondet_uint();
union V
{
  unsigned char c;
  unsigned int i;
};
int main()
{
  union
  {
    unsigned long long l;
    struct
    {
      union V v[1];
      char c[2];
    } s;
  } u;
  union V w;
  w.i = nondet_uint();
  u.s.v[0] = w;
  __ESBMC_assert(u.s.v[0].i != 0x01020304u, "show");
}
