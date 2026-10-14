// The quiet half of the same defect. `*"abc"` dereferences an array without
// reaching the encoder's int/ptr cast, so the unported arm did not abort here
// -- it returned a wrong verdict, which a crash-class test cannot catch.
int main(void)
{
  char c = *"abc";

  __ESBMC_assert(c == 'a', "*\"abc\" is the first character");
  return 0;
}
