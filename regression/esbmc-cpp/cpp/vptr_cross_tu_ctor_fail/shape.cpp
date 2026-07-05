#include "shape.h"
Shape::Shape(int id) : id_(id) {}
int Shape::kind() const { return 1; }
Circle::Circle(int id) : Shape(id) {}
int Circle::kind() const { return 2; }
