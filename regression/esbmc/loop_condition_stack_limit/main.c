int nd(void);
int main()
{
  while (nd())
    ;
  char big[12]; /* 96 bits: over a 90-bit limit */
  big[0] = 0;
  return big[0];
}
