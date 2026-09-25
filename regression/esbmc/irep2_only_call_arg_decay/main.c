/* take_ptr is declared, not defined: clang leaves the array-to-pointer decay
   for the adjuster rather than inserting it in the AST. */
void take_ptr(int *p);

int main(void)
{
  int a[3];
  a[0] = 1;
  take_ptr(a);
  return 0;
}
