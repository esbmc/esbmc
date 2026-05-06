// Reproducer for https://github.com/esbmc/esbmc/issues/4232
// Derived struct with bitfield base + switch on static_cast<enum>(field).
// Was crashing: assert(r) in to_solver_smt_ast due to wrong aggregate init.
#include <cstdint>

enum class E : uint8_t { A = 0 };

struct Base { uint8_t hi : 4; uint8_t lo : 4; };
struct Req : Base { uint8_t data[2]; };

extern "C" uint8_t nondet_u8();

int main()
{
    uint8_t idx = nondet_u8();
    __ESBMC_assume(idx >= 1);

    Req req{};
    req.data[1] = nondet_u8();

    switch (static_cast<E>(req.data[0])) {
        case E::A:
            __ESBMC_assert(idx < 1, "OOB");
            (void)req.data[1];
            break;
        default: break;
    }
    return 0;
}
