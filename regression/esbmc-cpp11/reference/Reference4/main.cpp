#include <assert.h>
#include <utility>

int main() 
{
    int a = 10;

    assert(std::move(a) == 10);

    int &&rref = std::move(a);

    assert(rref == 10);

    rref = 3;

    assert(rref == 3);

    return 0;
}
