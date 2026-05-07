// Function call: instrumentation visits each function independently
// (per-function k-path scope, #4325). Each function contributes its own
// branch witnesses; goal count adds up across callers and callees.
int helper(int z)
{
  if (z > 0)
    return z + 1;
  return -z;
}

int main()
{
  int a;
  if (a > 0)
    return helper(a);
  return helper(-a);
}
