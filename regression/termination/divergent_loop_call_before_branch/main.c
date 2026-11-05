/* Choosing t != 0 every iteration runs forever, but the call between the
 * nondet and the branch stops the divergence check from treating t as a
 * free choice. */
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
