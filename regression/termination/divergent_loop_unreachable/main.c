/* The loop could run forever, but init() always fails, so it is never
 * reached and every execution terminates. */
extern int __VERIFIER_nondet_int(void);

int init(void)
{
  return -1;
}

int main(void)
{
  int x = 0;
  if (init() != 0)
    return 1;
  while (__VERIFIER_nondet_int())
    x++;
  return x;
}
