/* --scalarize-structs (#7521): each field of a struct reached only by name
 * becomes a variable of its own; everything else keeps its struct. */
#include <assert.h>

struct inner
{
  int a;
  int b[2];
};

struct sample
{
  int total;
  char tag;
  struct inner in;
};

struct flex
{
  int n;
  int data[];
};

struct sample g = {1, 'g', {2, {3, 4}}};
struct sample table[2] = {{5, 't', {6, {7, 8}}}, {9, 'u', {10, {11, 12}}}};
extern struct inner ext;
struct flex fl;
int nondet_int(void);

int sum(struct sample s)
{
  return s.total + s.in.a + s.in.b[0] + s.in.b[1];
}

struct sample make(int v)
{
  struct sample r = {v, 'm', {v, {v, v}}};
  return r;
}

int main(void)
{
  g.total = g.total + 1;
  g.in.b[1] = g.in.a + 3;
  assert(sum(g) == 2 + 2 + 3 + 5);

  struct sample copy = g;
  assert(copy.in.b[1] == 5 && copy.tag == 'g');

  struct sample fresh;
  fresh.total = 7;
  assert(fresh.total == 7);

  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  struct sample picked;
  picked = table[i];
  assert(picked.total == 5 || picked.total == 9);

  struct sample swapped = {1, 's', {2, {0, 0}}};
  swapped = (struct sample){swapped.in.a, 's', {swapped.total, {0, 0}}};
  assert(swapped.total == 2 && swapped.in.a == 1);

  struct sample either = {1, 'e', {0, {0, 0}}};
  either = i ? either : copy;
  assert(either.total == 1 || either.total == 2);

  struct sample made = make(4);
  assert(made.in.b[1] == 4);

  struct sample aliased = {0, 'a', {0, {0, 0}}};
  int *p = &aliased.total;
  *p = 6;
  assert(aliased.total == 6);

  fl.n = 3;
  assert(fl.n == 3 && ext.a == ext.a);

  for (int k = 0; k < 3; k++)
  {
    struct inner scoped = {k, {k, k}};
    assert(scoped.a == k);
  }
  return 0;
}
