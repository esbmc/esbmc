#!/usr/bin/env python3

from testing_tool import (TestCase, apply_transform_over_tests, FAIL_MODES)

import sys
import os.path
import getopt
import shlex
import fnmatch
import operator

SOLVERS = [
    'smtlib',
    'z3',
    'boolector',
    'cvc',
    'mathsat',
    'yices',
    'bitwuzla'
]

EXTENSIONS = {
    'lang=c'     : {'c', 'i'},
    'lang=cxx'   : {'cc', 'cpp'},
    'lang=cuda'  : {'cu'},
    'lang=py'    : {'py'},
    'lang=sol'   : {'solast'},
    'lang=jimple': {'jimple'},
}

OPT2FLAGS = {
    '--' + solver: {'solver=' + solver} for solver in SOLVERS
} | { # Frontend related
    '--16'           : {'bitw=16'},
    '--32'           : {'bitw=32'},
    '--64'           : {'bitw=64'},
    '--binary'       : {'goto'},
    '--little-endian': {'endian=little'},
    '--big-endian'   : {'endian=big'},
    '--no-arch'      : {'endian=none', 'arch=none'},
    '--ppc-macos'    : {'arch=ppc', 'os=macos', 'bitw=32'},
    '--i386-macos'   : {'arch=x86', 'os=macos', 'bitw=32'},
    '--i386-linux'   : {'arch=x86', 'os=linux', 'bitw=32'},
    '--i386-win32'   : {'arch=x86', 'os=win', 'bitw=32'},
    '--cheri'        : {'cheri'},
    '--old-frontend' : {'old'},
} | { # Strategy related
    '--k-induction'    : {'strat=kind'},
    '--incremental-bmc': {'strat=incr'},
    '--falsification'  : {'strat=falsify'},
    '--termination'    : {'strat=term'},
} | { # Optimiziation related
    '--interval-analysis': {'ia'},
    '--gcse'             : {'gcse'},
}

BUG = 'bug'

FLAG_DESC = {}

class Flag(str):
    def __new__(cls, s):
        assert len(s) > 0
        assert s[0] not in '+-', "invalid flag: '%s'" % s
        assert all(c not in '()|&%' for c in s), "invalid flag: '%s'" % s
        return super().__new__(cls, s)

    @property
    def value(self) -> str:
        return self[self.find('=') + 1:]

    @property
    def category(self):
        eq = self.find('=')
        return None if eq < 0 else self[:eq]

    def matches(self, gpat_flag, smatch = fnmatch.fnmatchcase) -> bool:
        scat = self.category
        pcat = gpat_flag.category
        return ((pcat is None or scat is not None and smatch(scat, pcat)) and
                smatch(self.value, gpat_flag.value))

    def __repr__(self) -> str:
        return "Flag(%s)" % super().__repr__()

def known_flags() -> set[Flag]:
    flgs = {BUG} | EXTENSIONS.keys()
    for v in OPT2FLAGS.values():
        flgs |= v
    return {Flag(f) for f in flgs}

def list_flags(verbosity : int) -> None:
    # collect
    flgs = known_flags()
    # output
    if verbosity > 0:
        max_w = max(len(f) for f in flgs)
        for f in sorted(flgs):
            print('%*s: %s' % (-max_w, f, FLAG_DESC.get(f, '<undocumented>')))
    else:
        for f in sorted(flgs):
            print(f)

def flags(tc : TestCase) -> set[Flag]:
    r = set()
    if tc.test_mode in FAIL_MODES:
        r.add(Flag(BUG))
    for opt in tc.generate_run_argument_list('true')[1:]:
        f = set()
        if os.path.exists(opt) and os.path.isfile(os.path.realpath(opt)):
            # probably an input file
            ext = opt[opt.rfind('.') + 1:]
            for lang, exts in EXTENSIONS.items():
                if ext in exts:
                    f = {Flag(lang)}
                    break
        else:
            # not a file, so it's an option
            f = {Flag(g) for g in OPT2FLAGS.get(opt, f)}
        # add all the new flags to the ones we already accumulated
        r |= f
    return r

class Predicate:
    def __init__(self, fmt, op, *args):
        self.op = op
        self.args = args
        self.fmt = fmt

    def __and__(self, other):
        # Python operator '&'
        return Predicate('(%s & %s)', operator.and_, self, other)

    def __or__(self, other):
        # Python operator '|'
        return Predicate('(%s | %s)', operator.or_, self, other)

    def __invert__(self):
        # Python operator '~'
        return Predicate('!%s', operator.not_, self)

    def __call__(self, variables) -> bool:
        # Evaluation
        return self.op(*(arg(variables) for arg in self.args))

    def __str__(self):
        return self.fmt % tuple(str(arg) for arg in self.args)

class GlobPattern(Predicate):
    def __init__(self, qstr : str):
        super().__init__(qstr.replace('%', '%%'), None)
        self.gpat = Flag(qstr)

    def __call__(self, flgs : set[Flag]) -> bool:
        return any(f.matches(self.gpat) for f in flgs)

# if category is set, allow only one value
class Consistent(Predicate):
    def __init__(self):
        super().__init__('%%consistent', None)
        self.unique = {}
        for f in known_flags():
            fcat = f.category
            if fcat is not None:
                self.unique.setdefault(fcat, set()).add(f)

    def __call__(self, flags : set[Flag]):
        return all(len(flags & uniqs) <= 1 for uniqs in self.unique.values())

NAMED_PREDICATES = {str(pred): pred for pred in [
    Consistent(),
]}
assert all(k[0] == '%' for k in NAMED_PREDICATES)

def parse_formula(s : str, debug = False) -> Predicate:
    def Factor(s):
        if debug:
            print("fin '%s'" % s)
        if s[0] == '(':
            t, a = Formula(s[1:].strip())
            assert t and t[0] == ')', "expected ')', but got: '%s'" % t
            return t[1:].strip(), a
        if s[0] in '!-':
            t, a = Factor(s[1:].strip())
            return t, ~a
        def efind(s, c):
            idx = s.find(c)
            return len(s) if idx < 0 else idx
        term = min(efind(s, c) for c in ' ()!-&|')
        if debug:
            print("fin2 '%s', term: %d" % (s, term))
        assert term != 0
        t = s[term:].strip()
        s = s[:term]
        if debug:
            print("fin3 s: '%s', t: '%s'" % (s, t))
        if s[0] == '%':
            return t, NAMED_PREDICATES[s]
        return t, GlobPattern(s)

    def Product(s):
        if debug:
            print("And '%s'" % s)
        t, a = Factor(s)
        while t and t[0] not in ')|':
            if debug:
                print("And2 '%s'" % t)
            if t[0] == '&':
                t = t[1:].strip()
            t, b = Factor(t)
            a = a & b
        return t, a

    def Formula(s):
        if debug:
            print("Or '%s'" % s)
        t, a = Product(s)
        while t and t[0] == '|':
            t, b = Product(t[1:].strip())
            a = a | b
        return t, a

    t, a = Formula(s.strip())
    assert not t, "garbage after parsing formula: '%s'" % t
    return a

def usage() -> None:
    print('usage: %s [-OPTS] [--] [REG_PATH [REG_PATH [...]]]' %
          (sys.argv[0] if len(sys.argv) > 0 else 'regdb.py'))
    print('''
Options:
  -h          display this help message
  -l          list known flags
  -q QUERY    output only test cases satisfying QUERY, multiple queries can be
              performed and each will further restrict the list of results
  -v          increase verbosity (display flags for each test case or
              description for each flag), use twice to show shell-escaped
              cmdline options instead of flags

Multiple paths can optionally be specified as REG_PATH parameters, which will
restrict the test cases to process to only those given. If none are specified,
the default is to search the built-in list of all non-disabled regression tests.

QUERY is an infix formula (! or - for negation, & or space for conjunction, |
for disjunction) over glob patterns being matched against flags of test cases.
QUERY has this grammar (with spaces allowed around any of Formula, Product and
Factor):

  Formula   ::= Product ('|' Product)*
  Product   ::= Factor ('&'? Factor)*
  Factor    ::= '(' Formula ')'
              | ('!' | '-') Factor
              | '%' Predicate
              | GlobPattern
  Predicate ::= 'consistent'

The named predicate 'consistent' is true iff the set of flags of a test case are
consistent, that is, for categorized flags of the form CATEGORY=VALUE at most
one VALUE is allowed per CATEGORY.
'''[:-1])

def main():
    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'hlq:v')
    except getopt.GetoptError as err:
        return 'error: %s' % err

    verbosity = len(tuple(opt for opt, val in opts if opt == '-v'))

    if ('-h', '') in opts:
        return usage()

    if ('-l', '') in opts:
        return list_flags(verbosity)

    tcs = []
    if args:
        tcs.extend(TestCase(arg, os.path.basename(arg)) for arg in args)
    else:
        apply_transform_over_tests(tcs.append)

    queries = [parse_formula(val) for opt, val in opts if opt == '-q']
    if queries:
        q = queries[0]
        for i in range(1, len(queries)):
            q = q & queries[i]
        tcs = [tc for tc in tcs if q(flags(tc))]

    if len(tcs) == 0 or verbosity == 0:
        # just directory
        for tc in tcs:
            print(tc.test_dir)
    else:
        # directory + flags or cmdline options
        max_w = max(len(tc.test_dir) for tc in tcs)
        for tc in tcs:
            value = (' '.join(sorted(flags(tc))) if verbosity == 1 else
                     shlex.join(tc.generate_run_argument_list('true')[1:]))
            print('%*s: %s' % (-max_w, tc.test_dir, value))

if __name__ == '__main__':
    sys.exit(main())
