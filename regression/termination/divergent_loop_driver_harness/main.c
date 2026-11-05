/* The LDV driver-harness shape: the loop may only exit when the nondet value
 * is 0 and the device is closed, so always choosing to stay runs forever,
 * whichever handler each iteration calls. */
extern int __VERIFIER_nondet_int(void);

int opened;

int dev_open(void)
{
  if (__VERIFIER_nondet_int())
    return -1;
  opened = 1;
  return 0;
}

void dev_close(void)
{
  opened = 0;
}

int main(void)
{
  int tmp, op;
  if (__VERIFIER_nondet_int())
    return 0;
  while (1)
  {
    tmp = __VERIFIER_nondet_int();
    if (!tmp && !opened)
      break;
    op = __VERIFIER_nondet_int();
    if (op == 0)
      dev_open();
    else if (op == 1)
      dev_close();
  }
  return 0;
}
