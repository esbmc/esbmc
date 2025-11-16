# Regular Expression Operational Model
# TODO: Currently, the regex model uses manual pattern recognizers with
# nondeterministic fallbacks. A proper string solver would handle
# regex operations formally and completely. We'll have to consider the following:
# 1) We will need to represent strings as first-class types in our intermediate representation.
# 2) We will need to integrate existing string solvers such as Z3-str, CVC5, or Z3's sequence theory.
# 3) String constraint solving can be expensive; we may need further strategies for handling large string programs.


def try_match_char_class_range(pattern: str, pattern_len: int, string: str) -> int:
    """Match [x-y]+ or [x-y]* patterns"""
    if pattern_len != 6:
        return -1

    # Extract pattern characters to avoid constant folding issues
    ch0: str = pattern[0]
    ch2: str = pattern[2]
    ch4: str = pattern[4]
    if not (ch0 == '[' and ch2 == '-' and ch4 == ']'):
        return -1

    quantifier: str = pattern[5]
    if quantifier != '+' and quantifier != '*':
        return -1

    start_char: str = pattern[1]
    end_char: str = pattern[3]
    string_len: int = len(string)

    if string_len == 0:
        return 1 if quantifier == '*' else 0

    i: int = 0
    while i < string_len:
        c: str = string[i]
        if c < start_char or c > end_char:
            return 0
        i = i + 1
    return 1


def try_match_digit_sequence(pattern: str, pattern_len: int, string: str) -> int:
    """Match \d+ or \d* patterns"""
    if pattern_len != 3:
        return -1

    # Extract pattern characters to avoid constant folding issues
    ch0: str = pattern[0]
    ch1: str = pattern[1]
    if ch0 != '\\' or ch1 != 'd':
        return -1

    quantifier: str = pattern[2]
    if quantifier != '+' and quantifier != '*':
        return -1

    string_len: int = len(string)

    if string_len == 0:
        return 1 if quantifier == '*' else 0

    i: int = 0
    while i < string_len:
        c: str = string[i]
        if c < '0' or c > '9':
            return 0
        i = i + 1

    return 1


def try_match_alternation(pattern: str, pattern_len: int, string: str) -> int:
    """Match (x|y)z* patterns"""
    if pattern_len != 7:
        return -1

    # Extract pattern characters to avoid constant folding issues
    ch0: str = pattern[0]
    ch2: str = pattern[2]
    ch4: str = pattern[4]
    ch6: str = pattern[6]
    if not (ch0 == '(' and ch2 == '|' and ch4 == ')' and ch6 == '*'):
        return -1

    option1: str = pattern[1]
    option2: str = pattern[3]

    if len(string) == 0:
        return 0

    if string[0] == option1 or string[0] == option2:
        return 1
    return 0


def try_match_two_char_class_range(pattern: str, pattern_len: int, string: str) -> int:
    """Match ^[x-y][x-y]$ patterns (exactly two characters in range)"""
    if pattern_len != 12:
        return -1

    # Extract pattern characters to avoid constant folding issues
    ch0: str = pattern[0]
    ch1: str = pattern[1]
    ch3: str = pattern[3]
    ch5: str = pattern[5]
    ch6: str = pattern[6]
    ch8: str = pattern[8]
    ch10: str = pattern[10]
    ch11: str = pattern[11]
    if not (ch0 == '^' and ch1 == '[' and ch3 == '-' and ch5 == ']' and
            ch6 == '[' and ch8 == '-' and ch10 == ']' and ch11 == '$'):
        return -1

    start_char1: str = pattern[2]
    end_char1: str = pattern[4]
    start_char2: str = pattern[7]
    end_char2: str = pattern[9]

    string_len: int = len(string)

    # Must be exactly 2 characters
    if string_len != 2:
        return 0

    # Check both characters are in their respective ranges
    if string[0] < start_char1 or string[0] > end_char1:
        return 0
    if string[1] < start_char2 or string[1] > end_char2:
        return 0

    return 1


def try_match_dot_literal(pattern: str, pattern_len: int, string: str) -> int:
    """Match .x patterns (any character followed by literal character)"""
    if pattern_len != 2:
        return -1

    # Extract pattern character to avoid constant folding issues
    ch0: str = pattern[0]
    if ch0 != '.':
        return -1

    # Second character should be a literal (not a metacharacter)
    literal_char: str = pattern[1]
    if (literal_char == '.' or literal_char == '*' or literal_char == '+' or literal_char == '?'
            or literal_char == '[' or literal_char == ']' or literal_char == '('
            or literal_char == ')' or literal_char == '|' or literal_char == '^'
            or literal_char == '$' or literal_char == '\\'):
        return -1

    string_len: int = len(string)

    # Must be at least 2 characters
    if string_len < 2:
        return 0

    # Second character must match the literal
    if string[1] != literal_char:
        return 0

    return 1


def match(pattern: str, string: str) -> bool:
    """
    Try to match pattern at the beginning of string
    Returns True if successful, False otherwise    
    """
    pattern_len: int = len(pattern)

    # Empty pattern
    if pattern_len <= 1:
        return True

    # Universal match ".*"
    if pattern_len >= 2:
        # Extract pattern characters to avoid constant folding issues
        ch0: str = pattern[0]
        ch1: str = pattern[1]
        if ch0 == '.' and ch1 == '*':
            return True

    # Try each pattern recognizer
    result: int = try_match_char_class_range(pattern, pattern_len, string)
    if result >= 0:
        return result == 1

    result = try_match_digit_sequence(pattern, pattern_len, string)
    if result >= 0:
        return result == 1

    result = try_match_alternation(pattern, pattern_len, string)
    if result >= 0:
        return result == 1

    result = try_match_two_char_class_range(pattern, pattern_len, string)
    if result >= 0:
        return result == 1

    result = try_match_dot_literal(pattern, pattern_len, string)
    if result >= 0:
        return result == 1

    # Pattern ending with ".*"
    if pattern_len >= 3:
        # Extract pattern characters to avoid constant folding issues
        ch_last2: str = pattern[pattern_len - 2]
        ch_last1: str = pattern[pattern_len - 1]
        if ch_last2 == '.' and ch_last1 == '*':
            prefix_len: int = pattern_len - 2
            has_meta_in_prefix: bool = False
            i: int = 0
            while i < prefix_len:
                c: str = pattern[i]
                if c == '.' or c == '*' or c == '+' or c == '?' or c == '[' or c == ']' or c == '(' or c == ')' or c == '|' or c == '^' or c == '$' or c == '\\':
                    has_meta_in_prefix = True
                    break
                i = i + 1

            if not has_meta_in_prefix:
                if len(string) < prefix_len:
                    return False
                j: int = 0
                while j < prefix_len:
                    if pattern[j] != string[j]:
                        return False
                    j = j + 1
                return True

    # Check for metacharacters
    has_meta: bool = False
    k: int = 0
    while k < pattern_len - 1:
        ch: str = pattern[k]
        if ch == '.' or ch == '*' or ch == '+' or ch == '?' or ch == '[' or ch == ']' or ch == '(' or ch == ')' or ch == '|' or ch == '^' or ch == '$' or ch == '\\':
            has_meta = True
            break
        k = k + 1

    # Literal pattern matching
    if not has_meta:
        effective_pattern_len: int = pattern_len
        string_len: int = len(string)
        if string_len < effective_pattern_len:
            return False
        m: int = 0
        while m < effective_pattern_len:
            if pattern[m] != string[m]:
                return False
            m = m + 1
        return True

    # Nondeterministic fallback
    has_match: bool = __VERIFIER_nondet_bool()
    return has_match


def search(pattern: str, string: str) -> bool:
    """
    Search for pattern anywhere in string
    Returns True if found, False otherwise
    """

    # ".*" pattern always matches
    if len(pattern) >= 2:
        # Extract pattern characters to avoid constant folding issues
        ch0: str = pattern[0]
        ch1: str = pattern[1]
        if ch0 == '.' and ch1 == '*':
            return True

    pattern_len: int = len(pattern)

    # Check if pattern contains regex metacharacters
    has_meta: bool = False
    i: int = 0
    while i < pattern_len:
        c: str = pattern[i]
        if c == '.' or c == '*' or c == '+' or c == '?' or c == '[' or c == ']' or c == '(' or c == ')' or c == '|' or c == '^' or c == '$' or c == '\\':
            has_meta = True
            break
        i = i + 1

    # For literal patterns, check if pattern appears anywhere in string
    if not has_meta:
        if pattern_len > len(string):
            return False

        # Try matching at each position in string
        pos: int = 0
        while pos <= len(string) - pattern_len:
            match_found: bool = True
            j: int = 0
            while j < pattern_len:
                if string[pos + j] != pattern[j]:
                    match_found = False
                    break
                j = j + 1
            if match_found:
                return True
            pos = pos + 1
        return False

    # For patterns with metacharacters, use nondeterministic behavior
    has_match: bool = __VERIFIER_nondet_bool()
    return has_match


def fullmatch(pattern: str, string: str) -> bool:
    """
    Match pattern against entire string
    Returns True if entire string matches, False otherwise
    """

    # ".*" pattern matches everything
    if len(pattern) >= 2:
        # Extract pattern characters to avoid constant folding issues
        ch0: str = pattern[0]
        ch1: str = pattern[1]
        if ch0 == '.' and ch1 == '*':
            return True

    pattern_len: int = len(pattern)

    # Try each pattern recognizer (they already match entire string)
    result: int = try_match_char_class_range(pattern, pattern_len, string)
    if result >= 0:
        return result == 1

    result = try_match_digit_sequence(pattern, pattern_len, string)
    if result >= 0:
        return result == 1

    result = try_match_alternation(pattern, pattern_len, string)
    if result >= 0:
        return result == 1

    result = try_match_two_char_class_range(pattern, pattern_len, string)
    if result >= 0:
        return result == 1

    result = try_match_dot_literal(pattern, pattern_len, string)
    if result >= 0:
        return result == 1

    # Check if pattern contains regex metacharacters
    has_meta: bool = False
    i: int = 0
    while i < pattern_len:
        c: str = pattern[i]
        if c == '.' or c == '*' or c == '+' or c == '?' or c == '[' or c == ']' or c == '(' or c == ')' or c == '|' or c == '^' or c == '$' or c == '\\':
            has_meta = True
            break
        i = i + 1

    # For literal patterns, check if pattern exactly matches entire string
    if not has_meta:
        if len(pattern) != len(string):
            return False

        j: int = 0
        while j < pattern_len:
            if pattern[j] != string[j]:
                return False
            j = j + 1
        return True

    # For patterns with metacharacters, use nondeterministic behavior
    has_match: bool = __VERIFIER_nondet_bool()
    return has_match
