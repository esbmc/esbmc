/* x = 0 holds on the first entry to the inner loop only: the outer
 * loop increments x before re-entering it. With x == 1 the inner loop
 * resets j forever, so the ranking check must not seed x == 0. */
int main(void)
{
  for (int x = 0; x < 2; x++)
  {
    int j = 0;
    while (j < 2)
    {
      j++;
      if (x == 1)
        j = 0;
    }
  }
  return 0;
}
