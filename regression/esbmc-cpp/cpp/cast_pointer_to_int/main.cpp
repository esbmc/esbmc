#include <cstdint>

class Test {
    public:
      Test(char *p)
      {
        ptr = p;
      }
      char *ptr;
  };
  
  int main()
  {
    char *tmp, tmp2='a';
    uintptr_t addr;
    tmp = &tmp2;
    Test test(tmp);
    // unsigned int x = (unsigned int)p; 
    // ERROR: cast from pointer to smaller type
    addr =  reinterpret_cast<uintptr_t>(test.ptr);
  }
