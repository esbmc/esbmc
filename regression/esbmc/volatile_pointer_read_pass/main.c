struct Flags
{
  volatile int changing;
  int stable;
};

volatile int value;
struct Flags flags = {0, 7};

int main(void)
{
  volatile int *pointer = &value;
  int snapshot = *pointer;
  int copy = snapshot;
  __ESBMC_assert(pointer == &value, "taking an address preserves the object");
  __ESBMC_assert(snapshot == copy, "a saved value stays stable");
  __ESBMC_assert(flags.stable == 7, "ordinary members stay stable");

  volatile struct Flags whole;
  volatile struct Flags *whole_pointer = &whole;
  int member_snapshot = whole_pointer->stable;
  int member_copy = member_snapshot;
  __ESBMC_assert(member_snapshot == member_copy, "a saved member stays stable");

  volatile struct Flags *absent = 0;
  int guarded = absent ? absent->changing : 7;
  __ESBMC_assert(guarded == 7, "an unevaluated read has no pointer check");
  int short_circuit = absent == 0 || absent->changing;
  __ESBMC_assert(short_circuit, "short circuit guards the volatile access");
}
