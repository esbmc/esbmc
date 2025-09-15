#include <cassert>
#include <valarray>
#include <iostream>

int main() 
{
    std::valarray<int> arr = {1, 2, 3, 4, 5};
    assert(arr[0] == 1);
    assert(arr[4] == 5);

    std::valarray<int> arr2 = arr + 1;
    assert(arr2[0] == 2);
    assert(arr2[4] == 6);

    std::valarray<int> arr3 = arr * 2;
    assert(arr3[1] == 4);
    assert(arr3[3] == 8);

    int sum = arr.sum();
    assert(sum == 15);

    std::slice sl(0, 3, 1);
    std::valarray<int> arr4 = arr[sl];
    assert(arr4.size() == 3);
    assert(arr4[0] == 1 && arr4[1] == 2 && arr4[2] == 3);

    std::valarray<bool> mask = (arr % 2) == 0;
    std::valarray<int> arr5 = arr[mask];
    assert(arr5.size() == 1); // should fail
    assert(arr5[0] == 2 && arr5[1] == 4);

    return 0;
}

