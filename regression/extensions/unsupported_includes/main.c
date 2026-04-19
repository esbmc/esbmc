#include <stdatomic.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <emmintrin.h>


void test_nonatomic_to_atomic(int n) {
    _Atomic int atomic_var = n; // Implicit conversion from int to _Atomic(int)
}

int main()
{

}
