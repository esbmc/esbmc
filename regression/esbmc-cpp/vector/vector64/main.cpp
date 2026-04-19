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
        Car("Toyota", 120),
        Car("Honda", 130),
        Car("Ford", 110)
    };

    std::vector<Car> sorted_cars = cars;
    std::sort(sorted_cars.begin(), sorted_cars.end(), compare_speed);

    assert(std::is_sorted(sorted_cars.begin(), sorted_cars.end(), compare_speed) && "List is not sorted correctly!");

    std::cout << "Assertion passed successfully!" << std::endl;
    return 0;
}
