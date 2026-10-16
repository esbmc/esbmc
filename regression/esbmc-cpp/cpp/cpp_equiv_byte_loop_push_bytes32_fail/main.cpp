// C++ isolation test: byte-loop preservation push for bytes32.
//
// Mirrors the Solidity frontend's CURRENT lowering for `bytes32[]` state
// vars going through the generic _ESBMC_array_push helper at
// src/c2goto/library/solidity/solidity_array.c:313-363 (realloc + explicit
// __builtin_memcpy with size_t element size — per-byte preservation chain).
//
// Paired with cpp_equiv_typed_slot_push_bytes32_fail/, which uses the
// proposed typed-slot single-store. Shape parity vs Solidity
// regression/esbmc-solidity/napp_state_2d_dyn_bytes32_fail confirmed via
// --show-goto-functions memcpy/realloc call-site grep counts.
//
// The two C++ tests share the same planted bug (4th push absent but the
// assertion expects it on the freshest slot) so both produce VERIFICATION
// FAILED through the same VCC. The cost delta between them isolates the
// per-byte preservation chain from everything else.

#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <cstring>

struct bytes32_t { uint8_t data[32]; };

struct dyn_b32 {
    bytes32_t *data;
    size_t len;
    dyn_b32() : data(nullptr), len(0) {}

    // Generic byte-loop push: realloc the entire block (header + all prior
    // elements + new slot), then memcpy the new element in. Mirrors
    // _ESBMC_array_push's preservation chain at solidity_array.c:339-362.
    void push(const bytes32_t &e) {
        bytes32_t *nd = (bytes32_t *)realloc(data, (len + 1) * sizeof(bytes32_t));
        if (nd == nullptr) abort();
        data = nd;
        __builtin_memcpy(&data[len], &e, sizeof(bytes32_t));
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
        assert(v.size() == 3);
        assert(v[2].data[31] == 0xAC);
        // Planted bug: 4th push omitted, assertion expects it.
        assert(v.size() >= 4 && v[3].data[31] == 0xDD);
    }
};

int main() {
    C c;
    c.run();
    return 0;
}
