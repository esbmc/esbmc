// q->f writes i through a pointer, so it must forget i's lower bound 1:
// i * 3 overflows on the sixth iteration.
int __VERIFIER_nondet_int(void);

struct S
{
  int f;
};

int main()
{
  int n = __VERIFIER_nondet_int();
  if (n < 0)
    return 0;
  int a = 0;
  int i = 1;
  struct S *q = (struct S *)&i;
  while (i <= n / 3)
  {
    a = i * 3;
    if (i == 5)
      q->f = -2147483647;
    else
      i = i + 1;
  }
  return a;
}
