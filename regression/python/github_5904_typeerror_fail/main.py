# Regression for issue #5904: `int + str` must raise a catchable TypeError that
# escapes main() (previously the frontend crashed with a JSON type_error).
def main():
    y = 1 + "s"


main()
