// Real: update always sees x == 1 ((unsigned char)257 == 1), a climbs to 4,
// exit. Every execution terminates.
int a = 0;
extern int __VERIFIER_nondet_int(void);
extern void exit(int);
int update(int x)
{
  if (x == 257)
  {
    a = 1 - a;
    return 1;
  }
  if (x == 1)
  {
    a = a + 1;
    if (a > 3)
      exit(0);
    return 2;
  }
  exit(0);
  return -2;
}
int main(void)
{
  int sink = 0;
  while (1)
  {
    int input;
    input = __VERIFIER_nondet_int();
    if (input != 1 && input != 257)
      return -2;
    sink = update((unsigned char)input);
  }
  return sink;
}
