typedef void (*handler)(void);

unsigned long nondet_ulong(void);

int main()
{
  // Nothing in the program can supply this target.
  handler h = (handler)nondet_ulong();
  h();
  return 0;
}
