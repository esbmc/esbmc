// #9650: the generated argv set-up must not leave line-0 waypoints.
extern int __VERIFIER_nondet_int(void);

int main(int argc, char *argv[])
{
  int x = __VERIFIER_nondet_int();
  if (x == 3)
    __ESBMC_assert(0, "reached");
  return 0;
}
