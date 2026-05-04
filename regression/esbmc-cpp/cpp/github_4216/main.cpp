// Reproducer for https://github.com/esbmc/esbmc/issues/4216
// Expected: VERIFICATION FAILED ("OOB")
// Was crashing: assert(a->sort->get_data_width() == b->sort->get_data_width()) in mk_eq
#include <cstdint>

enum class E : uint8_t { A = 0 };

struct Req { uint8_t data[2]; };

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
