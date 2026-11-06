// The loop's only entry is a conditional jump past its head, which the
// inductive step cannot havoc; unhavoced, it proved this overflow absent.
extern int __VERIFIER_nondet_int(void);
int main(void)
{
  int i = 0, x = 0, z = 0;
  if (__VERIFIER_nondet_int())
    goto mid;
  return 0;
top:
  z = 0;
mid:
  i = i + 1;
  if (i == 50)
    x = 2147483647;
  z = x + 1;
  if (i < 100)
    goto top;
  return z;
}
