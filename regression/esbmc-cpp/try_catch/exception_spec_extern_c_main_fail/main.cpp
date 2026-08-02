// An exception escaping the program entry point is uncaught and must be
// reported, regardless of how `main` is spelled. An `extern "C" int main()`
// mangles to the bare id `c:@F@main` (no parameter suffix), which the
// uncaught-escape epilogue check must still recognise as the entry function.
// Regression for a false negative where bare-`main` escaped entry detection
// and the throw was silently accepted as VERIFICATION SUCCESSFUL.
extern "C" int main()
{
  throw 42;
  return 0;
}
