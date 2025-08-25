score: int = 85
grade: str = "A" if score >= 90 else ("B" if score >= 80 else "C")
assert grade == "B"
