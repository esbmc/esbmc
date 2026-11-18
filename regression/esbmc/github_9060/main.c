/* #9060: an aggregate local leaves no member entries in the value set once
 * it is dead. */
struct S
{
  int *p;
  int *q;
};
int g, h;

void f(void)
{
  struct S s = {&g, &h};
  *s.p = 1;
}

int main(void)
{
  {
    struct S t = {&g, &h};
    *t.q = 2;
  }
  f();
  int *x = &g;
  __ESBMC_assert(*x == 1, "f wrote g");
  return 0;
}
