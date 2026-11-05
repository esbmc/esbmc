/* Two loops can each run forever, but only the second is reachable: the
 * witness must name that one. */
extern int __VERIFIER_nondet_int(void);

int fails(void)
{
  return 1;
}

int main(void)
{
  int x = 0;
  if (!fails())
    while (__VERIFIER_nondet_int())
      x++;
  while (__VERIFIER_nondet_int())
    x--;
  return 0;
}
