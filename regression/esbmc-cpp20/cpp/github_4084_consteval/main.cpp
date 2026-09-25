// C++20 consteval. Issue #4084 reported consteval and concepts as
// unsupported, but the failure was clang rejecting the source for want of a
// C++20 mode. Concepts already have coverage in github_4190_concept_combo;
// this pins the consteval half.
consteval int sq(int x)
{
  return x * x;
}

template <typename T>
concept Num = requires(T a) { a + a; };

template <Num T>
T twice(T v)
{
  return v + v;
}

int main()
{
  constexpr int k = sq(5);
  static_assert(k == 25);
  __ESBMC_assert(k == 25, "consteval result reaches the model");
  __ESBMC_assert(twice(3) == 6, "concept-constrained template instantiates");
  return 0;
}
