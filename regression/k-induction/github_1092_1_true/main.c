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

  // Kind algorithm struggles with this
  do
  {
    __ESBMC_assert(status != 3, "");
  } while(0);

  return 0;
}