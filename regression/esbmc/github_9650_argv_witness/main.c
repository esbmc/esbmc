// #9650: a safe main(argc, argv) still verifies.
extern int __VERIFIER_nondet_int(void);

int main(int argc, char *argv[])
{
  int x = __VERIFIER_nondet_int();
  if (x == 3)
    x = 4;
  return x == 3;
}
