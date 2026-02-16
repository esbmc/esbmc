def check_range(x: int, y: int) -> int:
    if x > 0 and y > 0:
        return 1
    else:
        return 0

# cover all decision outcomes
check_range(5, 3)   
check_range(5, -1)  
check_range(-1, 3)  
check_range(-1, -1) 