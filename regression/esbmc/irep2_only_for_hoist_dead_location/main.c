/* A non-block for-body leaves the hoisted wrapper block with no end location,
   so the DEAD unwound at its close is unlocated. */
int main(void)
{
  int s = 0;
  for (int i = 0; i < 3; i++)
    s = s + i;
  return s;
}
