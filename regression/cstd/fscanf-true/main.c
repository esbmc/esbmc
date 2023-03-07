#include <stdio.h>
void CWE191_Integer_Underflow__int64_t_fscanf_multiply_18_bad()
{
    __int64_t data;
    data = 0LL;
    goto source;
source:
    fscanf (stdin, "%" "l" "d", &data);
    goto sink;
sink:
    if(data < 0)
    {
	__ESBMC_assert(data < 0, "assertion should hold");
    }

}
int main(int argc, char * argv[])
{
    CWE191_Integer_Underflow__int64_t_fscanf_multiply_18_bad();
    return 0;
}
