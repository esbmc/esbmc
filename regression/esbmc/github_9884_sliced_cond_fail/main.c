// #9884: with --no-assertions the slicer removes the definition of cond, so
// the branch in check() must get no waypoint rather than an arbitrary one.
extern int __VERIFIER_nondet_int(void);
extern void
__assert_fail(const char *, const char *, unsigned int, const char *);

void check(int cond)
{
  if (!(cond))
    __assert_fail("0", "main.c", 10, "check");
}

int main(void)
{
  int x = __VERIFIER_nondet_int();
  check(x > 5);
  return x + 1;
}
