struct s
{
  unsigned long m;
  signed int t : 3;
};

struct u
{
  unsigned long m;
  unsigned int t : 3;
};

static struct s area[2];

int nondet_int(void);

int main(void)
{
  area[0].t = -3;
  area[1].t = 2;
  int i = nondet_int();
  __ESBMC_assume(i == 0 || i == 1);
  struct u *q = (struct u *)&area[i];
  __ESBMC_assert(q->t != 5, "area[0].t reads as 5");
  return 0;
}
