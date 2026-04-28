// KNOWNBUG: ESBMC's BuiltinBitCastExpr handler in clang_c_convert.cpp
// currently lowers __builtin_bit_cast via gen_typecast, which for two
// arithmetic types of the same size performs an arithmetic value cast
// (e.g. uint32_t 0xFFFFFFFF -> float 4.29e9) instead of a byte-level
// reinterpret (which would yield NaN). The round-trip therefore loses
// the original bit pattern. Tracked separately from #4191 (whose fix
// is the pointer specialisation in <bit>); promote to CORE once the
// non-pointer lowering is fixed.
#include <bit>
#include <cstdint>

extern "C" uint32_t __VERIFIER_nondet_uint32_t();

int main()
{
  uint32_t bits = __VERIFIER_nondet_uint32_t();
  float f = std::bit_cast<float>(bits);
  uint32_t round_trip = std::bit_cast<uint32_t>(f);
  __ESBMC_assert(round_trip == bits, "non-pointer bit_cast round-trip");
  return 0;
}
