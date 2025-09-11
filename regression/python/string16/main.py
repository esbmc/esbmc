def return_input(x: str) -> str:
    return x


def input_return(data: str) -> str:
    return data


assert return_input("keyword1") == "keyword1"
assert input_return("keyword2") == "keyword2"
