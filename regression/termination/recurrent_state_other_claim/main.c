// The frame exceeds --stack-limit, which fails a claim of its own in the run
// that looks for a recurrent state. That failure shows no repeated state.
int main(void)
{
  volatile char big[4096];
  big[0] = 0;
  unsigned x = 1;
  while (x != 0)
    x <<= 8;
  return big[0];
}
