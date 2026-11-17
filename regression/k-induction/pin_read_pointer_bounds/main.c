/* The loop only reads through p and reassigns it to b, so the inductive step
 * havocs p. Pinned to the objects it may point to, the dereference gets the
 * base case's bounds checks; a bare nondet pointer would point to an unknown
 * object, checked only for validity. */
int a[1];
int b[5];

int main()
{
  int *p = a;
  int s = 0;
  while (1)
  {
    if (__ESBMC_same_object(p, b))
    {
      s += *p;
      p++;
    }
    else
      p = b;
  }
}
