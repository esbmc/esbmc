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
  area[0].a = 7;
  area[1].a = 7;
  area[1].p = 1;
  int i = nondet_int();
  __ESBMC_assume(i == 0 || i == 1);
  struct s *q = &area[i];
  __ESBMC_assert(!q->p && q->a == 7, "area[1].p is set");
  return 0;
}
