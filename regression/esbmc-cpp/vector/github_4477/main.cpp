#include <vector>

class Car
{
public:
  int brand, speed;
  Car(int brand, int speed) : brand(brand), speed(speed) {}
};

static std::vector<int> make()
{
  std::vector<int> v;
  v.push_back(7);
  return v;
}

int main()
{
  // initializer_list construction (the issue's reproducer)
  std::vector<Car> cars = {Car(1, 120), Car(2, 130), Car(3, 110)};

  // simple local vector with growth
  std::vector<int> a;
  a.push_back(1);
  a.push_back(2);

  // copy construction (deep copy) followed by independent mutation
  std::vector<int> b = a;
  b.push_back(3);

  // copy assignment
  std::vector<int> c;
  c = a;

  // return by value
  std::vector<int> d = make();

  return cars[0].brand + a[0] + b[2] + c[1] + d[0];
}
