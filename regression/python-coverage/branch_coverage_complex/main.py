def grade_score(score: int) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

def process_grade(score: int, extra_credit: int) -> str:
    total: int = score + extra_credit
    if total > 100:
        total = 100
    
    grade: str = grade_score(total)
    
    if grade == "A" or grade == "B":
        if extra_credit > 0:
            return "Pass with honors"
        else:
            return "Pass"
    elif grade == "C" or grade == "D":
        return "Pass"
    else:
        return "Fail"

# Test all branches
process_grade(95, 0)   
process_grade(92, 5)  
process_grade(85, 0)   
process_grade(82, 3)   
process_grade(75, 0)   
process_grade(65, 0)   
process_grade(55, 0)   
process_grade(98, 10) 
