# A no-argument custom exception that is not caught propagates uncaught.
class E(Exception):
    pass
raise E()
