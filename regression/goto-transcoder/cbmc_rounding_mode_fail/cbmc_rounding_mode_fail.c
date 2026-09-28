int main()
{
  double a = 1.0, b = 3.0;
  __CPROVER_rounding_mode = 1; /* FE_DOWNWARD */
  double lo = a / b;
  __CPROVER_rounding_mode = 2; /* FE_UPWARD */
  double hi = a / b;
  __CPROVER_assert(lo == hi, "wrong: the modes agree");
  return 0;
}
