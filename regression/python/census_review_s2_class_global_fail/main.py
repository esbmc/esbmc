def f(divmod=None):

    class A:
        global divmod
        x = divmod(1, 0)


f()
