/* The first call binds the parameter; the second passes nothing, so the
   binding must not carry over. */
void f();

int main(void)
{
  f(7);
  f();
  return 0;
}

void f(int a)
{
  __ESBMC_assert(a == 7, "the second call passes no argument");
}
