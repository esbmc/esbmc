// Loop body with one branch; tests that prefix extends across
// unrolled iterations (Huang, Meyer & Weber, ICTSS 2025).
int main()
{
  int x;
  for (int i = 0; i < 3; i++)
  {
    if (x > 0)
      x = x - 1;
    else
      x = x + 1;
  }
  return x;
}
