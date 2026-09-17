/* C11 6.5.17p2: a comma expression has its right operand's type. Clang hands
   it the decayed pointer type when that operand is an array, so leaving it
   makes `(c, g[i])[0]` index a pointer rather than the row -- the named
   array-bounds check is replaced by the generic dereference one, and the
   printed subscript becomes (&g[i][0])[0]. Reduced from 00_aiob_4. */
unsigned g[42][3];
unsigned i;

int main(void)
{
  i = 3;
  if ((i = i, g[i])[0] != 0)
    return 1;
  return 0;
}
