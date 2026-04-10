def recursive_access(data:list, depth:int):
    if depth <= 0:
        return 0
    return data[depth] + recursive_access(data, depth - 1)

def main():
    numbers = [5, 10, 15, 20]
    result = recursive_access(numbers, 4)
    assert result > 0

main()
