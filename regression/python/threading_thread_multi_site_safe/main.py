import threading


result_a: int = 0
result_b: int = 0


def setter_a(x: int) -> None:
    global result_a
    result_a = x


def setter_b(x: int) -> None:
    global result_b
    result_b = x


def make_a() -> None:
    t = threading.Thread(target=setter_a, args=(7,))
    t.start()
    t.join()


def make_b() -> None:
    t = threading.Thread(target=setter_b, args=(9,))
    t.start()
    t.join()


make_a()
make_b()

# Each construction site must drive its OWN target — silent cross-scope
# site_id reuse would land both writes on the same global.
assert result_a == 7
assert result_b == 9
