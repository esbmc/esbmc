// C++ isolation test: memoised dynarr_idx alternative.
//
// Replaces the _ESBMC_dynarr_idx fold body with a single load from an
// SMT-array memo keyed on (addr<<32)|idx. Same clone-walker shape as
// cpp_equiv_dynarr_idx_xor_fold_pass/.
//
// EXPECTED OUTCOMES (recording the result is part of the deliverable):
//   (a) Post-slice SSA drops materially vs xor_fold baseline →
//       direction C2-A (memoise) is a GO candidate.
//   (b) Bitwuzla rejects with "Equality over constant arrays not fully
//       supported yet" → C2-A is NO-GO, reproduces Phase 1 finding.
//   (c) SSA delta < 5% → C2-A is NEEDS-MORE-DATA.
//
// SOUNDNESS NOTE — the memo is NOT a bijection over (addr,idx) → key.
// Sound only as a memo over an underlying injective fold, NOT as
// replacement. If this test measures C2-A as GO, the implementation
// proposal must combine fold + memo, not memo alone.

#include <cstdint>
#include <cassert>

extern "C" {
  unsigned __nondet_uint();
}

static uint64_t fold_cache[1 << 20];

static inline uint64_t dynarr_idx_memo(uint32_t addr, uint32_t idx) {
    // Shift-or pack: Phase 1 noted Bitwuzla rejects this pattern on
    // constant arrays. Recording the failure mode (if reproduced) is a
    // test deliverable.
    uint64_t key = ((uint64_t)addr << 32) | (uint64_t)idx;
    return fold_cache[key & 0xFFFFF];
}

#define N 8
static uint32_t state_arr[1 << 20];

void clone_walker(uint32_t base_addr, uint32_t clone_addr) {
    for (uint32_t i = 0; i < N; ++i) {
        uint64_t b_key = dynarr_idx_memo(base_addr,  i);
        uint64_t c_key = dynarr_idx_memo(clone_addr, i);
        state_arr[c_key & 0xFFFFF] = state_arr[b_key & 0xFFFFF];
    }
}

int main() {
    uint32_t base  = __nondet_uint();
    uint32_t clone = __nondet_uint();
    clone_walker(base, clone);
    return 0;
}
