/* p is rewritten through two 32-bit halves, so its value set still holds only
 * the NULL it was initialised with; p nonetheless points at x, so reading
 * through it is not a NULL dereference. */
int x = 7;

int main()
{
  int *p = 0;
  unsigned long a = (unsigned long)&x;
  ((unsigned *)&p)[0] = (unsigned)a;
  ((unsigned *)&p)[1] = (unsigned)(a >> 32);
  __ESBMC_assert(p == &x, "p is x");
  return *p;
}
