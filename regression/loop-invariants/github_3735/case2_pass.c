#define MAX 2

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

  // CASE 2: expr,fun; fun FIRST in invar: PASS
  qm = nondet_bool();
  qn = (qm + 1) % MAX;

  __ESBMC_loop_invariant(
    in_range( qn ) &&    // fun is FIRST
    0 <= qm && qm < 2 &&
    (qm != qn)
  );
  for (int count = 0; count < 10; count++)
  {
    qm = inc( qm );
    qn = inc( qn );
  }

  return 0;
}
