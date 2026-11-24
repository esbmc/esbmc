extern int nondet_int(void);

struct in
{
  int a;
  int b;
};

struct out
{
  struct in i;
  int c;
};

int main(void)
{
  struct out o;
  struct in t;
  t.a = nondet_int();
  t.b = 2;
  o.i = t;
  o.c = 1;
  __ESBMC_assert(o.i.a == t.a && o.i.b == 2, "whole write of a member");
  struct out q;
  q.i = o.i;
  q.c = 1;
  unsigned char *bytes = (unsigned char *)&q;
  bytes[0] = 7;
  __ESBMC_assert(q.i.a != 7 || q.c == 1, "byte write");
  return 0;
}
