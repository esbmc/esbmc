struct s
{
  unsigned long m;
  unsigned int a : 8;
  unsigned int p : 1;
};

static struct s area[2];

int nondet_int(void);

int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i == 0 || i == 1);
  struct s *q = &area[i];
  q->p = 1;
  __ESBMC_assert(area[1 - i].p == 1, "the store leaves area[1 - i].p clear");
  return 0;
}
