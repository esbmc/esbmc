// Reproducer for https://github.com/esbmc/esbmc/issues/4237
// Value-initializing a struct that inherits from a class via {} crashed ESBMC
// in the SMT encoding phase (to_solver_smt_ast assertion) due to a type
// mismatch in the flat aggregate-init path.
#include <cstdint>
extern "C" { uint8_t nondet_u8(); }

class Base {
    uint8_t _x{};
protected:
    void check(uint8_t i) { __ESBMC_assert(i < _x, "OOB"); }
public:
    Base() = default;
};

struct Derived : Base { using Base::check; };

int main() {
    Derived d{};
    uint8_t i = nondet_u8();
    __ESBMC_assume(i >= 4);
    d.check(i);
    return 0;
}
