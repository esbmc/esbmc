// Passing variant for https://github.com/esbmc/esbmc/issues/4232
// Same struct/enum pattern, but the assertion holds.
#include <cstdint>

enum class E : uint8_t { A = 0 };

struct Base { uint8_t hi : 4; uint8_t lo : 4; };
struct Req : Base { uint8_t data[2]; };

extern "C" uint8_t nondet_u8();

int main()
{
    Req req{};
    req.data[1] = nondet_u8();

    switch (static_cast<E>(req.data[0])) {
        case E::A:
            __ESBMC_assert(req.data[1] < 256, "always true");
            break;
        default: break;
    }
    return 0;
}
