// CWG 1457, which clang applies to C++98 too: 1 << 31 fits unsigned int, so it is defined.
int nondet_int();
int main()
{
  int k = nondet_int() & 31;
  int r = 1 << k;
  return r & 1;
}
