// 2 << 31 does not fit unsigned int: still an overflow before C++20.
int nondet_int();
int main()
{
  int k = nondet_int() & 31;
  int r = 2 << k;
  return r & 1;
}
