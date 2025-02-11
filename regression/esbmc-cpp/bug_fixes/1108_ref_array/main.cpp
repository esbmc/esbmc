// TC description:
//  Test function argument and return are both lvalue references.
//  This TC is a simplified version of esbmc-cpp/template/template_1 and _3

#include<cassert>

class FixedArray25
{
  public:
    int anValue[25];
};

// Returns a reference to the nIndex element of rArray
int& Value( FixedArray25 &rArray, int nIndex)
{
  return rArray.anValue[nIndex];
}

int main()
{
    FixedArray25 sMyArray;

    Value(sMyArray, 10) = 5;
    assert(sMyArray.anValue[10] == 5);
    Value(sMyArray, 15) = 10;
    assert(sMyArray.anValue[15] == 10);
    assert(sMyArray.anValue[10] == 5);
    //assert(sMyArray.anValue[10] == 10); // should be 5

    return 0;
}
