/* The loop havoc of k-induction picks variables by their names, and a member
 * name such as this one excluded a field variable named after it: the
 * inductive step then started from the value before the loop and proved the
 * assertion (#7521). */
struct S
{
  int my_pthread_lib_count;
  int y;
};

int main(void)
{
  struct S s = {0, 0};
  while (1)
  {
    s.my_pthread_lib_count++;
    __ESBMC_assert(s.my_pthread_lib_count < 3, "count stays small");
  }
}
