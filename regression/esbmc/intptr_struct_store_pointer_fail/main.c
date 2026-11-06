extern int __VERIFIER_nondet_int(void);
struct port
{
  unsigned int a0, a1, a2, a3;
  unsigned char b0, b1, b2, b3;
  unsigned long c0, c1, c2, c3;
  void *d0, *d1, *d2, *d3;
};
static struct port ports[1];
int g, h;
int main(void)
{
  ports[0].d1 = &h;
  int i = __VERIFIER_nondet_int();
  if (i > 0)
    return 0;
  unsigned long base = (unsigned long)ports + i * sizeof(struct port);
  *((void **)(base + 64)) = (void *)&g;
  __ESBMC_assume(i == 0);
  *(int *)ports[0].d1 = 3;
  __ESBMC_assert(g == 0, "g was written: must fail");
  return 0;
}
