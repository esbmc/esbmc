def parse() -> tuple[str, str]:
    return (nondet_str(), nondet_str())


scheme, token = parse()
if scheme == "bearer":
    raise ValueError("unsupported scheme")
