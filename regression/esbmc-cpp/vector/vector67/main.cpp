#include <vector>

class Car {
public:
    int brand;
    int speed;

    Car(int brand, int speed) : brand(brand), speed(speed) {}
};

int main() {
    std::vector<Car> cars = {
        Car(1, 120),
        Car(2, 130),
        Car(3, 110)
    };

    return 0;
}
