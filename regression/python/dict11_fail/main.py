def test_all_strings():
    """Should PASS: Dictionary with only string values"""
    translations: dict = {
        'hello': 'hola',
        'goodbye': 'adios',
        'thanks': 'gracias'
    }
    
    assert translations['hello'] == 'hhola'
    assert translations['goodbye'] == 'addios'
    assert translations['thanks'] == 'graccias'

test_all_strings()
