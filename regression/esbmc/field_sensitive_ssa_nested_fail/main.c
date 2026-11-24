extern int nondet_int(void);

struct in
{
  int a;
  int b;
};

struct mid
{
  struct in x;
  long l;
};

struct out
{
  struct mid m;
  struct in i;
  int c;
};

static int sum(struct out o)
{
  return o.m.x.b + o.i.a;
}

int main(void)
{
  struct out o;
  o.m.x.a = nondet_int();
  o.m.x.b = 0;
  o.i.a = 3;
  o.c = 0;
  o.m.l = 0;
  for (int k = 0; k < 4; k++)
  {
    o.c += k;
    o.m.x.b++;
    o.i.b = nondet_int() ? o.m.x.a : k;
  }
  struct in t = o.i;
  struct out p = o;
  __ESBMC_assert(p.c == 6, "counter");
  __ESBMC_assert(t.a == 3, "copied member");
  __ESBMC_assert(p.m.x.b == 5, "nested counter");
  __ESBMC_assert(sum(o) == 7, "by value");
  return 0;
}
