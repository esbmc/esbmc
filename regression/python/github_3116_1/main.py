import toml as local_toml


def main() -> None:
    data = """
[servers]
[servers.alpha]
ip = "10.0.0.1"
role = "frontend"
"""

    assert local_toml.loads(data) == "foo"


main()
