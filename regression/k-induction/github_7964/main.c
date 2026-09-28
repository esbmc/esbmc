// #7964: the inductive step must havoc p, not pin it to its loop-entry value.
int main()
{
  int a[2] = {1, 2};
  int *p = a;
  int s = 0;
  for (int i = 0; i < 3; i++)
  {
    s += *p; // reads a[2] on the third iteration
    p++;
  }
  return s;
}
