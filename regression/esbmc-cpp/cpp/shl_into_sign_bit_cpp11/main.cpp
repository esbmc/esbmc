// C++11 [expr.shift]/2 (CWG 1457): 1 << 31 fits unsigned int, so it is defined.
int nondet_int();
int main()
{
  int k = nondet_int() & 31;
  int r = 1 << k;
  return r & 1;
}
