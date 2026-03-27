# Three subclasses all passed to the same function.
# Common ancestor of Square, Circle, Triangle is Shape.


class Shape:

    def area(self):
        return 0


class Square(Shape):

    def area(self):
        return 4


class Circle(Shape):

    def area(self):
        return 3


class Triangle(Shape):

    def area(self):
        return 2


def print_area(s):
    a = s.area()
    assert a >= 0


sq = Square()
ci = Circle()
tr = Triangle()

print_area(sq)
print_area(ci)
print_area(tr)
