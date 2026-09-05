/* The C++ half of the OTHER-in-the-body decline: `delete p` lowers to
 * `OTHER delete p`, which the schema's variable havoc cannot describe. The
 * recogniser must decline so this use-after-free is still reported. */
unsigned int nondet_uint();

int main()
{
  unsigned int n = nondet_uint();
  __ESBMC_assume(n >= 1 && n <= 4);

  int *p = new int(0);
  unsigned int i = 0;
  unsigned int s = 0;

  while (i < n)
  {
    delete p;
    s = s + 2;
    i = i + 1;
  }

  *p = 3;
  return 0;
}
