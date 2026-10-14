import threading


# Regression test: a module-top subclass binding and a function-scope
# Thread(target=...) binding sharing the same variable name must NOT
# share a trampoline. Before the per-scope precedence fix the subclass
# entry shadowed the function-local target= entry, causing helper()'s
# w.start() to spawn the subclass's no-op trampoline instead of writer
# — a false VERIFICATION SUCCESSFUL on a program that should detect the
# write.


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        # No-op: this subclass exists only to occupy the name `w` at
        # module scope.
        pass


x: int = 0


def writer() -> None:
    global x
    x = 100


# Module-top subclass binding.
w: Worker = Worker()


def helper() -> None:
    # Function-scope target= binding reusing the name `w`. The local
    # binding must win for this scope, so w.start() spawns `writer`.
    w = threading.Thread(target=writer)
    w.start()
    w.join()


helper()
# After helper(), writer ran, so x == 100. Reading x after helper()
# returns is sequenced after the local join, so the value is visible.
assert x == 100
