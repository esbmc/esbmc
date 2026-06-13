// Exercises C++-frontend constructs under --irep2-bodies (V.4.3, esbmc#4715):
// classes with constructors, methods returning by value (temporary_object +
// new_object migration), and functions with exception specifications
// (code_cpp_src_throw_decl migration path).

struct Point
{
  int x, y;
  Point(int a, int b) : x(a), y(b)
  {
  }
  int sum() const
  {
    return x + y;
  }
  // operator+ returns by value: exercises temporary_object + new_object paths.
  Point operator+(const Point &o) const
  {
    return Point(x + o.x, y + o.y);
  }
};

// noexcept exercises the source-level exception-spec (throw_decl) path.
int add(int a, int b) noexcept
{
  return a + b;
}

int main()
{
  Point p(3, 4);
  __ESBMC_assert(p.sum() == 7, "sum==7");
  Point q = p + Point(1, 2);
  __ESBMC_assert(q.sum() == 10, "sum==10");
  __ESBMC_assert(add(2, 3) == 5, "add==5");
  return 0;
}
