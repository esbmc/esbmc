// Unwinding destroys only the objects between the throw and the matching
// handler: an object in an enclosing scope outlives the catch.
struct Guard
{
  bool *f;
  Guard(bool *p) : f(p)
  {
  }
  ~Guard()
  {
    *f = true;
  }
};
struct E
{
};

int main()
{
  bool outer_destroyed = false, inner_destroyed = false;
  Guard outer(&outer_destroyed);
  try
  {
    Guard inner(&inner_destroyed);
    throw E();
  }
  catch (E &)
  {
    __ESBMC_assert(
      inner_destroyed && !outer_destroyed,
      "inner destroyed, outer still alive");
  }
  return 0;
}
