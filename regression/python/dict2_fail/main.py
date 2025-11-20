def test_basic_dict():
    dict = {'name':'Bob', 'ref':'Python', 'sys':'Win'}
    assert dict['ref'] == 'pPython'
    assert dict['name'] == 'bBob'
    assert dict['sys'] == 'wWin'

test_basic_dict()
