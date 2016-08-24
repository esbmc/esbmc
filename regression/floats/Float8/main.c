int main()
{
  double d, q, r;
  __ESBMC_assume(__ESBMC_isfinited(q));
  d=q;
  r=d+0;
  assert(r==d);
}
