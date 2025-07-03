#include <assert.h>

int main()
{
  if( -((unsigned int)-1) < 0U){
    assert(0);
  }
  return 0;
}
