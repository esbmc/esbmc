def foo(
    s: str,
    t: str,
    u: str,
    v: str,
) -> None:
    invalid_chars: list[str] = ['v']
    for char in invalid_chars:
        assert char not in s
    
    if t is not None:
        l2 = ['banana_split', 'elephant_trunk', 'sunshine_valley', 'rainbow_colors', 
                     'mountain_range', 'ocean_waves', 'butterfly_wings', 'universe_stars', 'penguin_colony']
        assert t in l2

    if u is not None:
        l3 = ['ABCDE', 'FGHIJ', 'KLMN', 'OPQRSTU']
        assert u in l3

    if v is not None:
        l4 = ['BUTTERFLY', 'MOONLIGHT', 'STARLIGHT', 'RAINBOW', 
                               'OCEAN_BREEZE', 'SUNSET', 'MOUNTAIN_DEW', 'FIREFLY', 'STARDUST']

        assert v in l4

foo('hello', 'banana_ssplit', 'ABCDE', 'BUTTERFLY')
