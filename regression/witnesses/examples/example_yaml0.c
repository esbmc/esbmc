void reach_error()
{
}
extern unsigned char __VERIFIER_nondet_uchar(void);
int main()
{
  unsigned char n = __VERIFIER_nondet_uchar();
  if (n == 0)
  {
    return 0;
  }
  unsigned char v = 0;
  unsigned int s = 0;
  unsigned int i = 0;
  while (i < n)
  {
    v = __VERIFIER_nondet_uchar();
    s += v;
    ++i;
  }
  if (s < v)
  {
    reach_error();
    return 1;
  }
  if (s > 65025)
  {
    reach_error();
    return 1;
  }
  return 0;
}
