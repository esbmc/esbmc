// Second C++ translation unit. Its body must survive the mergeASTs import so
// that multi_tu_second() deterministically returns 2 (not a nondet value).
extern "C" int multi_tu_second()
{
  return 2;
}
