#include <assert.h>
#include <float.h>

int main() 
{
    double subnormal1 = DBL_TRUE_MIN * 5;
    double subnormal2 = DBL_TRUE_MIN * 4;
    double result = subnormal1 - subnormal2;
    
    assert(result == DBL_TRUE_MIN); // Should equal DBL_TRUE_MIN
    return 0;
}
