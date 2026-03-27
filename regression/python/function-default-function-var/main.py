def multiply(x:int,y:int) -> int:
    return x*y

def subtract(x:int,y:int) -> int:
    return x - y

current_op = multiply

def do_op(operand:int, operator = current_op)->int:
    return operator(operand,2)

assert do_op(2) == 4

current_op = subtract

assert do_op(2) == 4
