int status = 0;
int foo()
{
  // status = *
  while(__VERIFIER_nondet_int()) // __verifier_nondet_int$1 = *
  {
    if(!status) // status = *
    {
      status = 1;
    }
    else if(status == 1)
    {
      status = 2;
    }
  } // status = 1 U 2
}

int bar()
{
  do
  {
    __ESBMC_assert(status != 3, "");
  } while(0);
}
int main()
{
  foo();
  bar();
  return 0;
}