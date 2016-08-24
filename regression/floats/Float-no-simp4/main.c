// all classification

int main()
{
  double d1, _d1;
  d1=_d1;
  __ESBMC_assume(__ESBMC_isnormald(d1));
  assert(!__ESBMC_isnand(d1));
  assert(!__ESBMC_isinfd(d1));
  assert(__ESBMC_isfinited(d1));

  double d2, _d2;
  d2=_d2;
  __ESBMC_assume(__ESBMC_isinfd(d2));
  assert(!__ESBMC_isnormald(d2));
  assert(!__ESBMC_isnand(d2));

  double d3, _d3;
  d3=_d3;
  __ESBMC_assume(__ESBMC_isnand(d3));
  assert(!__ESBMC_isnormald(d3));
  assert(!__ESBMC_isinfd(d3));
  assert(d3!=d3);

  double d4, _d4;
  d4=_d4;
  __ESBMC_assume(__ESBMC_isfinited(d4));
  assert(!__ESBMC_isnand(d4));
  assert(!__ESBMC_isinfd(d4));

  double d5, _d5;
  d5=_d5;
  __ESBMC_assume(!__ESBMC_isnand(d5) && !__ESBMC_isinfd(d5));
  assert(__ESBMC_isfinited(d5));
}
