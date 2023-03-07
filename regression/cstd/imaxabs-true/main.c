#include <math.h>
#include <stdio.h>
int main(int argc, char * argv[])
{
    int64_t data;
    data = 0LL;
    fscanf (stdin, "%" "l" "d", &data);
    if (data > (-0x7fffffffffffffff - 1) && imaxabs((intmax_t)data) <= sqrtl(0x7fffffffffffffffLL))
    {
        int64_t result = data * data;
        printLongLongLine(result);
    }
    else
    {
        printLine("data value is too large to perform arithmetic safely.");
    }
}
