/* A doubling arithmetic chain flattens to 2^N leaves under
 * canonicalization; the node cap must keep the run fast and the
 * branch must still verify. */
unsigned int nondet_u(void);

int main(void)
{
  unsigned int a = nondet_u();
  unsigned int c = a;
  for (int i = 0; i < 40; i++)
    c = c + c;
  if ((c & 1u) == 0u)
    __ESBMC_assert((c & 1u) == 0u, "doubled value is even");
  return 0;
}
