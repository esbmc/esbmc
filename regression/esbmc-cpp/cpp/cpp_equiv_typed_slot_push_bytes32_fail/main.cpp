// C++ isolation test: typed-slot push for bytes32 (PROPOSED fast path).
//
// Mirrors a hypothetical extension of _ESBMC_array_push_uint256
// (src/c2goto/library/solidity/solidity_array.c:383-421) to bytes32:
// fresh malloc + typed single-assign on slot[old_len]; stale slots
// become nondet on the new allocation. Sound over-approximation when
// push is the only mutator and reads target the freshest slot.
//
// Paired with cpp_equiv_byte_loop_push_bytes32_fail/. The cost delta
// between the two tests on identical {--incremental-bmc, --unwind 5}
// budget projects the Stage 1.a saving for bytes32. The planted bug is
// on the freshest slot, immune to stale-nondet, so both tests produce
// VERIFICATION FAILED via the same VCC.

#include <cstdint>
#include <cassert>
#include <cstdlib>

struct bytes32_t { uint8_t data[32]; };

struct dyn_b32 {
    bytes32_t *data;
    size_t len;
    dyn_b32() : data(nullptr), len(0) {}

    // Typed-slot push: fresh malloc, typed single store on the new
    // slot. No preservation — stale slots become nondet on the new
    // allocation, which is sound when push is the only mutator and
    // reads target the freshest slot.
    void push(const bytes32_t &e) {
        bytes32_t *nd = (bytes32_t *)malloc((len + 1) * sizeof(bytes32_t));
        if (nd == nullptr) abort();
        nd[len] = e;                  // typed single store
        data = nd;
        len++;
    }
    size_t size() const { return len; }
    bytes32_t &operator[](size_t i) { return data[i]; }
};

class C {
public:
    dyn_b32 v;
    void run() {
        for (uint8_t k = 0; k < 3; ++k) {
            bytes32_t b{};
            b.data[31] = (uint8_t)(0xAA + k);
            v.push(b);
        }
        // NOTE: do not assert v[0..1] under typed-slot semantics — those
        // slots are stale-nondet after the third push. Reading the
        // freshest slot v[2] is the only legitimate per-slot read.
        assert(v.size() == 3);
        assert(v[2].data[31] == 0xAC);
        // Planted bug on the freshest slot — immune to stale-nondet.
        assert(v.size() >= 4 && v[3].data[31] == 0xDD);
    }
};

int main() {
    C c;
    c.run();
    return 0;
}
