// The inner loop sees j = 0 at its head on every entry. That is no cycle, as
// the state saved on one entry must not be compared on the next.
int main(void)
{
  int a[3] = {1, 1, 0};
  for (int x = 0; x < 3; x++)
  {
    int j = 0;
    while (a[j] != 0)
      j++;
  }
  return 0;
}
