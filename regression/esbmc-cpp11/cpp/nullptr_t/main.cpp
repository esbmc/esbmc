#include <iostream>

void myFunction(int* ptr) {
    if (ptr == nullptr) {
        std::cout << "Pointer is nullptr" << std::endl;
    } else {
        std::cout << "Pointer is not nullptr" << std::endl;
    }
}

int main() {
    int* ptr1 = nullptr;
    int* ptr2 = new int(5);

    myFunction(ptr1); // Passing nullptr
    myFunction(ptr2); // Passing valid pointer

    delete ptr2;

    return 0;
}
