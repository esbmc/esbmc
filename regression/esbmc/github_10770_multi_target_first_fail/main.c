// Each target of p gets a step at the store, holding the same nondet in its
// struct literal; only the call's one value is named.
extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);

struct pair
{
  int a, b;
};

struct pair g1, g2;

int main(void)
{
  struct pair *p = __VERIFIER_nondet_int() ? &g1 : &g2;
  p->b = __VERIFIER_nondet_int();
  if (p == &g1 && g1.b == 3)
    reach_error();
  return 0;
}
