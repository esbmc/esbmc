/* #10080: the only float in the branch is 0x3DCCCCCE; a 6-digit input reads
 * back as 0x3DCCCCCD and misses it. */
extern void reach_error(void);
extern float __VERIFIER_nondet_float(void);
int main()
{
  float x = __VERIFIER_nondet_float();
  if (x > 0.1f && x < 0.10000002f)
    reach_error();
  return 0;
}
