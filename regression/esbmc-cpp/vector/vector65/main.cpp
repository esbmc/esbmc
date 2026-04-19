#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>

class Car {
public:
    std::string brand;
    int speed;

    Car(std::string brand, int speed) : brand(brand), speed(speed) {}
};

bool compare_speed(const Car& c1, const Car& c2) {
    return c1.speed < c2.speed;
}

int main() {
    std::vector<Car> cars = {
        Car("T", 200),
        Car("H", 130),
        Car("F", 110)
    };

    std::vector<Car> sorted_cars = cars;
    // do not sort to make the assertion to fail!
    //std::sort(sorted_cars.begin(), sorted_cars.end(), compare_speed);

    assert(std::is_sorted(sorted_cars.begin(), sorted_cars.end(), compare_speed) && "List is not sorted correctly!");

    std::cout << "Assertion should fail!" << std::endl;
    return 0;
}
