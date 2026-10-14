// A recursive function that fully terminates within the unwind bound. No
// unwinding assertion is reached, so verification succeeds and no CWE is
// emitted. Guards against the uncontrolled-recursion detector ever firing
// on a well-behaved, terminating recursion.
int f(int n)
{
  if (n <= 0)
    return 0;
  return f(n - 1);
}

int main(void)
{
  return f(3);
}
