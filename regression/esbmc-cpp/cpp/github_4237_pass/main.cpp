// Value-init `Derived d{}` must zero-initialize the inherited _x field.
#include <cstdint>
extern "C" { uint8_t nondet_u8(); }

class Base {
    uint8_t _x{};
protected:
    void check() { __ESBMC_assert(_x == 0, "zero-init"); }
public:
    Base() = default;
};

struct Derived : Base { using Base::check; };

int main() {
    Derived d{};
    d.check();
    return 0;
}
