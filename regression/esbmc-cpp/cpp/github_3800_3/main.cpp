// Chain: A depends on B, B depends on C - all class statics
// defined in the order C, B, A (correct) but declared as A, B, C in class body
struct Chain
{
  static const int A;
  static const int B;
  static const int C;
};

// Define C first, then B (depends on C), then A (depends on B)
constexpr int Chain::C = 5;
constexpr int Chain::B = Chain::C + 3;
constexpr int Chain::A = Chain::B + 2;

int main()
{
  __ESBMC_assert(Chain::C == 5, "C");
  __ESBMC_assert(Chain::B == 8, "B");
  __ESBMC_assert(Chain::A == 10, "A");
  return 0;
}
