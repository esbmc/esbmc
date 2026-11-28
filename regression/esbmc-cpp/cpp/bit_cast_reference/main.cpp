// __builtin_bit_cast of a call that returns a reference reinterprets the
// referenced object, not the reference.
struct pair_t
{
  unsigned short lo, hi;
};

static float g = 1.0f;
static float &ref()
{
  return g;
}
static float &&rref()
{
  return static_cast<float &&>(g);
}
static pair_t make()
{
  return pair_t{0x0000, 0x3f80};
}

struct wide_t
{
  unsigned lo, hi;
};

int main()
{
  __ESBMC_assert(__builtin_bit_cast(unsigned, ref()) == 0x3f800000u, "lvalue ref");
  __ESBMC_assert(__builtin_bit_cast(unsigned, rref()) == 0x3f800000u, "rvalue ref");
  __ESBMC_assert(__builtin_bit_cast(unsigned, make()) == 0x3f800000u, "class prvalue");
  float named = 1.0f;
  __ESBMC_assert(__builtin_bit_cast(unsigned, named) == 0x3f800000u, "named lvalue");
  wide_t w{0x3f800000u, 1u};
  __ESBMC_assert(
    __builtin_bit_cast(unsigned long long, w) == 0x000000013f800000ull,
    "class to unsigned long long");
  return 0;
}
