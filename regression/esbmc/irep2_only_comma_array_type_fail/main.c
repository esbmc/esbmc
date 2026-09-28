/* Out of bounds through the same comma subscript: the row index must still be
   checked against the array's own bound, not merely as a dereference. */
unsigned g[42][3];
unsigned i;

int main(void)
{
  i = 50;
  if ((i = i, g[i])[0] != 0)
    return 1;
  return 0;
}
