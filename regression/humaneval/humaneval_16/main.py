# HumanEval/16
#esbmc: /home/weiqi/ESBMC_Project/esbmc/src/irep2/irep2_expr.h:2939: index2t::index2t(const type2tc&, const expr2tc&, const expr2tc&): Assertion `is_array_type(source) || is_vector_type(source)' failed.
#Aborted (core dumped)


def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct characters (regardless of case) does it consist of
    >>> count_distinct_characters('xyzXYZ')
    3
    >>> count_distinct_characters('Jerry')
    4
    """
    return len(set(string.lower()))

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert count_distinct_characters('') == 0
    # assert count_distinct_characters('abcde') == 5
    # assert count_distinct_characters('abcde' + 'cade' + 'CADE') == 5
    # assert count_distinct_characters('aaaaAAAAaaaa') == 1
    # assert count_distinct_characters('Jerry jERRY JeRRRY') == 5
    print("✅ HumanEval/16 - All assertions completed!")
