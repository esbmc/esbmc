int nondet_int(void);

int main(void)
{
  int value = nondet_int();
  if (value > 0)
    return 1;
  return 0;
}
