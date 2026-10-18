// C++ isolation test: current _ESBMC_dynarr_idx XOR fold in a
// clone-walker shape.
//
// Verbatim port of the fold at
// src/c2goto/library/solidity/solidity_array.c:31-62 (truncated to
// 128-bit for portability; the upper-bit fold rounds are zero under the
// `addr<2^32` invariant guaranteed by _ESBMC_get_unique_address).
//
// Shape mirrors the constructor clone walker at
// src/solidity-frontend/solidity_convert_constructor.cpp:1046-1115 —
// two fold calls per loop iteration (base address, clone address).
//
// PASS oracle: this is the cost baseline. Paired with
// cpp_equiv_dynarr_idx_memoised_pass/ which substitutes the fold body
// with a single SMT-array select; the cost delta isolates the fold
// arithmetic chain.

#include <cstdint>
#include <cassert>

extern "C" {
  unsigned __nondet_uint();
}

typedef unsigned __int128 uint256_lo_t;

static inline uint64_t dynarr_idx(uint256_lo_t addr, uint256_lo_t idx) {
    // Verbatim port of _ESBMC_dynarr_idx (solidity_array.c:59-61).
    uint64_t a = (uint64_t)addr ^ (uint64_t)(addr >> 32) ^
                 (uint64_t)(addr >> 64) ^ (uint64_t)(addr >> 96);
    uint64_t i = (uint64_t)idx  ^ (uint64_t)(idx  >> 64);
    return (a * 0x100000001ULL) ^ i;
}

#define N 8
static uint32_t state_arr[1 << 20];

void clone_walker(uint256_lo_t base_addr, uint256_lo_t clone_addr) {
    for (uint256_lo_t i = 0; i < N; ++i) {
        uint64_t b_key = dynarr_idx(base_addr,  i);
        uint64_t c_key = dynarr_idx(clone_addr, i);
        state_arr[c_key & 0xFFFFF] = state_arr[b_key & 0xFFFFF];
    }
}

int main() {
    uint256_lo_t base  = __nondet_uint() & 0xFFFFFFFFu;
    uint256_lo_t clone = __nondet_uint() & 0xFFFFFFFFu;
    clone_walker(base, clone);
    return 0;
}
