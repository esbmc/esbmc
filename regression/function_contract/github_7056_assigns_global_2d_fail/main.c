/* The element-wise frame check descends into the row, so a write to a column
 * of a row the clause does not name is caught -- comparing only the first
 * flattened element of each row would miss it. */
int m[3][4];

void f(int i, int v)
{
  __ESBMC_requires(i >= 0 && i < 3);
  __ESBMC_assigns(m[i]);
  __ESBMC_ensures(1);
  m[i][0] = v;
  m[0][3] = v; /* m[0] is only in assigns when i == 0 */
}

int main()
{
  return 0;
}
