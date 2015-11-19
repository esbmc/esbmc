#include <assert.h>
 
static_assert(sizeof(long) == 4, "Code relies on int being exactly 4 bytes");
 
int main(void)
{
    return 0;
}
