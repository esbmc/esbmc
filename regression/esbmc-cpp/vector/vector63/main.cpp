#include <vector>

class Car {
public:
    std::string brand;
    int speed;

    Car(std::string brand, int speed) : brand(brand), speed(speed) {}
};

int main() {
    std::vector<Car> cars = {
        Car("Toyota", 120),
        Car("Honda", 130),
        Car("Ford", 110)
    };

    return 0;
}
