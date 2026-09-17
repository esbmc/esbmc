int main() {
  double d = 1.5;
  __CPROVER_assert(__CPROVER_isnormald(d), "1.5 is normal");
  double z = 0.0;
  __CPROVER_assert(!__CPROVER_isnormald(z), "0.0 is not normal");
  return 0;
}
