// https://github.com/esbmc/esbmc/issues/11061
bool equal( int a, int b )
{
  return a == b;
}
int main()
{
  int idx = 0;
  int a, b;
  __ESBMC_loop_invariant( (idx == 0)  || equal(a, b) );
  __ESBMC_loop_invariant( 0 <= idx && idx <= 3 );
  while (idx < 3)
  {
    ++idx;
  }
}
