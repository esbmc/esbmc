/* Reading t is undefined, not a free choice of the program's, so the loop
 * is not taken for one that can run forever. */
int main(void)
{
  while (1)
  {
    int t;
    if (t)
      continue;
    break;
  }
  return 0;
}
