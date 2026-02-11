# Common join test via split + filter
if __name__ == "__main__":
    assert " ".join("a b".split(" ")) == "a b"
    assert "".join("".split(" ")) == ""
    print("ok")
