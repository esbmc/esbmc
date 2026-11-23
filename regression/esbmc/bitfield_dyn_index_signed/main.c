struct s
{
  unsigned long m;
  signed int t : 3;
};

static struct s area[2];

int nondet_int(void);

int main(void)
{
  area[0].t = -3;
  area[1].t = 2;
  int i = nondet_int();
  __ESBMC_assume(i == 0 || i == 1);
  struct s *q = &area[i];
  __ESBMC_assert(q->t == (i == 0 ? -3 : 2), "the signed bit-field keeps its value");
  return 0;
}
