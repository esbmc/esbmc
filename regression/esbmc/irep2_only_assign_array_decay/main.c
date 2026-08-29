/* A separate assignment statement, not an initialiser: clang inserts the decay
   for `int *p = a;` itself, but leaves `p = a;` to the adjuster. */
int main(void)
{
  int a[3];
  int *p;
  a[0] = 7;
  p = a;
  return p[0];
}
