class Animal:
    pass


class Dog(Animal):
    pass


class Car:
    pass


def main() -> None:
    pet: Animal = Dog()
    not_a_pet: Animal = Car()


main()
