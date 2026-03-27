def test_all_strings():
    """Should PASS: Dictionary with only string values"""
    translations: dict = {'hello': 'hola', 'goodbye': 'adios', 'thanks': 'gracias'}

    assert translations['hello'] == 'hola'
    assert translations['goodbye'] == 'adios'
    assert translations['thanks'] == 'gracias'


test_all_strings()
