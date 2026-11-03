# Issue #8225: an uncaught TypeError from divmod on a list escapes main().
def main():
    q, r = divmod([], 9)


main()
