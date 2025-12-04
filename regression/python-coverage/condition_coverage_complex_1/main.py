def validate_access(age: int, has_ticket: int, is_member: int, is_vip: int) -> int:
    if (is_member == 1 or(age >= 18 and has_ticket) ) and is_vip == 0:
        return 1 
    else:
        return 0  

def check_special_conditions(temp: int, humidity: int, wind: int) -> int:
    if (temp > 30 or (temp > 20 and humidity > 70)) and wind < 50:
        return 1  
    else:
        return 0  

# Cover all decision paths for validate_access

validate_access(20, 1, 0, 0)   
validate_access(20, 0, 1, 0)   
validate_access(20, 0, 0, 1)   
validate_access(15, 0, 0, 0)   
validate_access(20, 1, 0, 1)   
validate_access(15, 0, 1, 0)   

# Cover all decision paths for check_special_conditions
check_special_conditions(35, 50, 30)  
check_special_conditions(25, 80, 30)   
check_special_conditions(25, 60, 30)   
check_special_conditions(15, 80, 30)   
check_special_conditions(35, 50, 60)   