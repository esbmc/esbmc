#include <iostream>
#include <type_traits>

// Function template that takes two arguments and returns their sum
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
add(T a, T b) {
    return a + b;
}

int main() {
    // Call the add function with integer arguments
    std::cout << "add(3, 4): " << add(3, 4) << std::endl;

    // Uncommenting this line would result in a compile-time error
    //std::cout << "add(3.5, 4.5): " << add(3.5, 4.5) << std::endl;

    return 0;
}
