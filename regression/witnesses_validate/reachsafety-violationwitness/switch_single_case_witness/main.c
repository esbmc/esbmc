extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);
int main()
{
  int v = __VERIFIER_nondet_int();
  int r = 0;
  switch (v)
  {
  case 2:
    r = 2;
  }
  if (r == 2)
    reach_error();
  return 0;
}
