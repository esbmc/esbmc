# Regression for issue #5903: when the exception is caught, the type-partitioned
# uncaught-exception property must not fire.
def main():
    try:
        a = [1, 2, 3]
        z = a[5]
    except IndexError:
        pass


main()
