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
  // CASE 1: expr,fun; fun middle in invar: FAIL (bug case)

  qm = nondet_bool();
  qn = (qm + 1) % MAX; // expr for asn

  __ESBMC_loop_invariant(
    0 <= qm && qm < 2 && // reporter says this part is missing
    in_range( qn ) &&    // fun in MIDDLE
    (qm != qn)
  );
  for (int count = 0; count < 10; count++)
  {
    qm = inc( qm );
    qn = inc( qn );
  }

  //--------------------------------------------------
  // CASE 2: expr,fun; fun first in invar: PASS

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

  //--------------------------------------------------
  // CASE 3: expr,fun; fun last in invar: PASS

  qm = nondet_bool();
  qn = (qm + 1) % MAX;

  __ESBMC_loop_invariant(
    0 <= qm && qm < 2 &&
    (qm != qn) &&
    in_range( qn )       // fun is LAST
  );
  for (int count = 0; count < 10; count++)
  {
    qm = inc( qm );
    qn = inc( qn );
  }

  return 0;
}
