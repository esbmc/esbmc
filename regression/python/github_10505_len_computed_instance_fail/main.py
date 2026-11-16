# len() of an expression that evaluates to a class instance (#10505)
import array

assert len(array.array("b", [1, 2]) * 2) == 3
