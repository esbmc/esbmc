extern int __VERIFIER_nondet_int(void);
struct port
{
  unsigned int a0, a1, a2, a3;
  unsigned char b0, b1, b2, b3;
  unsigned long c0, c1, c2, c3;
  void *d0, *d1, *d2, *d3;
};
static struct port ports[1];
int g;
int main(void)
{
  int i = __VERIFIER_nondet_int();
  if (i > 0)
    return 0;
  unsigned long base = (unsigned long)ports + i * sizeof(struct port);
  *((unsigned int *)(base + 8)) = 5U;
  *((void **)(base + 40)) = (void *)&g;
  __ESBMC_assume(i == 0);
  __ESBMC_assert(ports[0].a2 == 5U, "store lands on a2");
  __ESBMC_assert(ports[0].c2 == (unsigned long)&g, "pointer store lands on c2");
  return 0;
}
