def check_range(x: int, y: int) -> int:
    if x > 0:
        if y > 0:
            return 1   
        else:
            return 4  
    else:
        if y > 0:
            return 2 
        else:
            return 3   


result: int = check_range(__VERIFIER_nondet_int(), __VERIFIER_nondet_int())