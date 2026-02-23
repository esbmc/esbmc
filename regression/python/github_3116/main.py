import toml  # type: ignore[import]


def main() -> None:
    toml_string = """
[servers]
[servers.alpha]
ip = "10.0.0.1"
role = "frontend"
"""

    parsed_toml = toml.loads(toml_string)
    assert parsed_toml == "foo"


main()