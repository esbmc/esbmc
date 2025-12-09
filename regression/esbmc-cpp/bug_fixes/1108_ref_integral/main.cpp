// TC description:
//  Test function argument and return are both lvalue references.
//  This TC is a simplified version of esbmc-cpp/template/template_2

#include<cassert>

int& Value1(int &value)
{
  return value;
}

int main()
{
    int x=5;
    assert(Value1(x)==5);
    return 0;
}
