# Regular Expression Operational Model


def match(pattern: str, string: str) -> bool:
    """
    Try to match pattern at the beginning of string
    Returns True if successful, False otherwise    
    """

    # Note: ESBMC strings include null terminator in length
    pattern_len: int = len(pattern)

    # Empty pattern (only null terminator)
    if pattern_len <= 1:
        return True  # Empty pattern matches at position 0

    # ".*" pattern matches everything
    if pattern_len >= 3:  # ".*" + null terminator
        if pattern[0] == '.' and pattern[1] == '*':
            return True

    # Simple character class patterns [a-z]+, [A-Z]+, [0-9]+
    if pattern_len == 7:  # "[x-y]+" + null = 7 chars
        if pattern[0] == '[' and pattern[2] == '-' and pattern[4] == ']' and pattern[5] == '+':
            start_char: str = pattern[1]
            end_char: str = pattern[3]
            string_len: int = len(string) - 1  # Exclude null terminator
            if string_len == 0:
                return False
            i: int = 0
            while i < string_len:
                c: str = string[i]
                if c < start_char or c > end_char:
                    return False
                i = i + 1
            return True
        # Handle [x-y]* patterns (zero or more)
        if pattern[0] == '[' and pattern[2] == '-' and pattern[4] == ']' and pattern[5] == '*':
            start_char: str = pattern[1]
            end_char: str = pattern[3]
            string_len: int = len(string) - 1  # Exclude null terminator
            if string_len == 0:
                return True  # Empty string matches with *
            i: int = 0
            while i < string_len:
                c: str = string[i]
                if c < start_char or c > end_char:
                    return False
                i = i + 1
            return True

    # case: (x|y)z* patterns
    if pattern_len == 8:  # "(x|y)z*" + null = 8 chars
        if (pattern[0] == '(' and pattern[2] == '|' and pattern[4] == ')' and 
            pattern[6] == '*'):
            option1: str = pattern[1]
            option2: str = pattern[3]
            
            if len(string) == 0:
                return False
            
            # First character must match one of the options
            if string[0] == option1 or string[0] == option2:
                return True
            return False

    # Pattern ending with ".*" (e.g., "a.*", "abc.*")
    if pattern_len >= 4:  # Need at least one char + ".*" + null terminator
        if pattern[pattern_len - 3] == '.' and pattern[pattern_len - 2] == '*':
            prefix_len: int = pattern_len - 3

            # Check if the prefix has metacharacters
            has_meta_in_prefix: bool = False
            i: int = 0
            while i < prefix_len:
                c: str = pattern[i]
                if c == '.' or c == '*' or c == '+' or c == '?' or c == '[' or c == ']' or c == '(' or c == ')' or c == '|' or c == '^' or c == '$' or c == '\\':
                    has_meta_in_prefix = True
                    break
                i = i + 1

            # If prefix is literal, check if string starts with it
            if not has_meta_in_prefix:
                if len(string) < prefix_len:
                    return False
                j: int = 0
                while j < prefix_len:
                    if pattern[j] != string[j]:
                        return False
                    j = j + 1
                return True

    # Check if pattern contains regex metacharacters (excluding null terminator)
    has_meta: bool = False
    k: int = 0
    while k < pattern_len - 1:  # Exclude null terminator
        ch: str = pattern[k]
        if ch == '.' or ch == '*' or ch == '+' or ch == '?' or ch == '[' or ch == ']' or ch == '(' or ch == ')' or ch == '|' or ch == '^' or ch == '$' or ch == '\\':
            has_meta = True
            break
        k = k + 1

    # For literal patterns, check if string starts with pattern
    if not has_meta:
        # Adjust pattern_len to exclude null terminator for comparison
        effective_pattern_len: int = pattern_len - 1
        if len(string) - 1 < effective_pattern_len:  # string length also includes null
            return False
        m: int = 0
        while m < effective_pattern_len:
            if pattern[m] != string[m]:
                return False
            m = m + 1
        return True

    # For patterns with metacharacters, use nondeterministic behavior
    has_match: bool = __VERIFIER_nondet_bool()
    return has_match

def search(pattern: str, string: str) -> bool:
    """
    Search for pattern anywhere in string
    Returns True if found, False otherwise
    """

    # ".*" pattern always matches
    if len(pattern) >= 2:
        if pattern[0] == '.' and pattern[1] == '*':
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
        if pattern[0] == '.' and pattern[1] == '*':
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