// C++ equivalent of regression/esbmc-solidity/napp_struct_multifield_fail
//
// Same shape: struct with 3 nested-array fields, identical push/pop/length
// pattern, dispatcher loop, single FAIL assertion (`flags[1].size() == 3`
// when actual size is 2). Mimics Solidity's modeling at the IR level:
// - state-var nested dynamic arrays via realloc
// - bytes32-like fixed-size element via 32-byte struct
// - bytes32[3]-like fixed-inner via 96-byte struct
// - dispatcher harness via while(__nondet_bool()) { run(); }
//
// Purpose: test whether ESBMC's backend (symex + SAT solver) can solve
// this shape when fed through the C++ frontend instead of the Solidity
// frontend. If C++ verifies fast and Solidity hangs, the gap is in
// Solidity-specific frontend lowering (e.g. _ESBMC_array_push library
// helper, sol_dynarray_state encoding, _sol_per_tx_reseed cost), not in
// the backend.
//
// Expected: VERIFICATION FAILED (the planted bug at
// `assert(flags[1].size() == 3)` should fire at the first dispatcher
// iteration since `flags[1]` only sees 2 pushes).

#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <cstring>

extern "C" {
  bool __nondet_bool();
  unsigned __nondet_uint();
}

// 32-byte struct mimicking bytes32 / address layout in Solidity model.
struct bytes32_t {
    uint8_t data[32];
};

// 96-byte struct mimicking bytes32[3] (fixed-inner, dyn-outer field).
struct hash_row_t {
    bytes32_t row[3];
};

// Manual realloc-based dynamic vector — mirrors Solidity's
// _ESBMC_array_push behavior (realloc + memcpy of new element).
template <typename T>
struct dyn_vec {
    T *data;
    size_t len;

    dyn_vec() : data(nullptr), len(0) {}

    void push(const T &elem) {
        T *nd = (T *)realloc(data, (len + 1) * sizeof(T));
        // Match _ESBMC_array_push's __ESBMC_assume(new_block != 0)
        if (nd == nullptr) abort();
        data = nd;
        memcpy(&data[len], &elem, sizeof(T));
        len++;
    }

    void pop() {
        if (len == 0) abort();
        len--;
    }

    T &operator[](size_t i) { return data[i]; }
    const T &operator[](size_t i) const { return data[i]; }

    size_t size() const { return len; }
};

struct Trio {
    dyn_vec<dyn_vec<uint64_t>> addrs;        // address[][] — 2D dyn-dyn
    dyn_vec<hash_row_t> hashes;              // bytes32[3][] — fixed-inner dyn-outer
    dyn_vec<bool> flags[2];                  // bool[][2] — dyn-inner fixed-2-outer
};

class C {
public:
    Trio trio;

    void pushAddrRow() {
        dyn_vec<uint64_t> empty;
        trio.addrs.push(empty);
    }

    void pushAddr(size_t r, uint64_t a) {
        trio.addrs[r].push(a);
    }

    void pushHashes(uint64_t a, uint64_t b, uint64_t c) {
        hash_row_t row = {};
        row.row[0].data[31] = (uint8_t)a;
        row.row[1].data[31] = (uint8_t)b;
        row.row[2].data[31] = (uint8_t)c;
        trio.hashes.push(row);
    }

    void pushFlag(size_t outer, bool v) {
        trio.flags[outer].push(v);
    }

    void run() {
        pushAddrRow();
        pushAddr(0, 0xAA);
        pushAddr(0, 0xBB);
        assert(trio.addrs.size() == 1);
        assert(trio.addrs[0].size() == 2);

        pushHashes(1, 2, 3);
        pushHashes(4, 5, 6);
        assert(trio.hashes.size() == 2);

        pushFlag(0, true);
        pushFlag(0, false);
        pushFlag(1, true);
        assert(trio.flags[0].size() == 2);
        assert(trio.flags[1].size() == 1);

        trio.addrs[0].pop();
        trio.hashes.pop();
        trio.flags[0].pop();

        pushAddr(0, 0xCC);
        pushHashes(9, 9, 9);
        pushFlag(0, false);
        pushFlag(1, false);

        // FLIPPED: trio.flags[1] has length 2 after second push, not 3
        assert(trio.flags[1].size() == 3);
    }
};

int main() {
    C contract;
    while (__nondet_bool()) {
        contract.run();
    }
    return 0;
}
