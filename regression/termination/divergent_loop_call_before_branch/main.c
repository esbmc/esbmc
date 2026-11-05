/* A call between the nondet and the branch that tests it. */
extern int __VERIFIER_nondet_int(void);
int calls;

void count(void)
{
  calls++;
}

int main(void)
{
  while (1)
  {
    int t = __VERIFIER_nondet_int();
    count();
    if (!t)
      break;
  }
  return 0;
}
