#include <cassert>

int main() {
    int arr[] = {1, 2, 3, 4, 5};

    for (int index = 1; int num : arr)
    {
        assert(num != index);
        index++;
    }

    return 0;
}
