def test_all_strings():
    translations: dict = {
        'hello': 'hola',
        'goodbye': 'adios',
        'thanks': 'gracias'
    }
    
    assert translations['hello'] == 'hhola'
    assert translations['goodbye'] == 'addios'
    assert translations['thanks'] == 'graccias'

test_all_strings()
