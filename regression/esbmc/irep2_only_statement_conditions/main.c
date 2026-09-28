/* goto_convert's branch lowering wants a boolean guard; clang leaves an integer
   controlling expression as it is written. */
int nondet_int(void);

int main(void)
{
  int a = nondet_int();
  int b = nondet_int();
  int c = nondet_int();
  int n = 0;

  if (a)
    n++;

  while (b)
    b = 0;

  for (; c;)
    c = 0;

  return n;
}
