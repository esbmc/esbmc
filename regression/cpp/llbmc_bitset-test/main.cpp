#include <cassert>
#include <bitset>
#include <iostream>

int main ()
{
    std::bitset<100> bs;

    for (int i = 0; i < 100; ++i) {
        if (i % 2) {
            bs[i] = true;
        }
    }

    std::bitset<100> bsf = bs;
    bsf.flip();

    assert(bs.count() == bsf.count());

    for (int i = 0; i < 100; ++i) {
        if (!(i % 2)) {
            assert(!bs.test(i));
            assert(bsf.test(i));
        } else {
            assert(bs.test(i));
            assert(!bsf.test(i));
        }
    }

    return 0;
}
