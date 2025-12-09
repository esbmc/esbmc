int main()
{
  int status = 0;
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

  while(nondet_int())
    __ESBMC_assert(status != 2, "");

  return 0;
}