#include <assert.h>

#define MAX 2

int mk()
{
  int res = nondet_int();
  res = res < 0 ? -res : res;
  res %= MAX;
  return res;
}

int inc( int qa )
{
  int res = (qa + 1) % MAX;
  return res;
}

int in_range( const int qa )
{
  return 0 <= qa && qa < MAX;
}

int main()
{
  int qm;
  int qn;

  //--------------------------------------------------
  // expr,fun; fun middle in invar: FAIL (bug case)

  qm = nondet_bool();
  qn = (qm + 1) % MAX; // expr for asn

  __ESBMC_loop_invariant(
    0 <= qm && qm < 2 && // this part is missing from invariant goto code
    in_range( qn ) && // fun for range, fun in middle
    (qm != qn)
  );
  for (int count = 0; count < 10; count++)
  {
    qm = inc( qm );
    qn = inc( qn );
  }

  return 0;
}
