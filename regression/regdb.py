#!/usr/bin/env python3

from testing_tool import (TestCase, apply_transform_over_tests, FAIL_MODES)

import sys
import os.path
import getopt
import shlex

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
    'c'     : {'c', 'i'},
    'cxx'   : {'cc', 'cpp'},
    'cuda'  : {'cuda'},
    'py'    : {'py'},
    'sol'   : {'solast'},
    'jimple': {'jimple'},
}

OPT2FLAGS = {
    '--' + solver: {solver} for solver in SOLVERS
} | { # Frontend related
    '--16'           : {'16'},
    '--32'           : {'32'},
    '--64'           : {'64'},
    '--binary'       : {'goto'},
    '--little-endian': {'le'},
    '--big-endian'   : {'be'},
    '--no-arch'      : {'ne', 'no-arch'},
    '--ppc-macos'    : {'ppc', 'macos', '32'},
    '--i386-macos'   : {'x86', 'macos', '32'},
    '--i386-linux'   : {'x86', 'linux', '32'},
    '--i386-win32'   : {'x86', 'win', '32'},
    '--cheri'        : {'cheri'},
    '--old-frontend' : {'old'},
} | { # Strategy related
    '--interval-analysis': {'ia'},
}

FLAG_DESC = {}

def list_flags(verbose : bool):
    # collect
    flgs = {'bug'} | EXTENSIONS.keys()
    for v in OPT2FLAGS.values():
        flgs |= v
    # output
    if verbose:
        max_w = max(len(f) for f in flgs)
        for f in sorted(flgs):
            print('%*s: %s' % (-max_w, f, FLAG_DESC.get(f, '<undocumented>')))
    else:
        for f in sorted(flgs):
            print(f)
    # finish
    sys.exit(0)

def flags(tc : TestCase):
    r = set()
    if tc.test_mode in FAIL_MODES:
        r.add('bug')
    opts = tc.generate_run_argument_list('true')[1:]
    for opt in opts:
        f = set()
        if os.path.exists(opt) and os.path.isfile(os.path.realpath(opt)):
            # probably an input file
            ext = opt[opt.rfind('.') + 1:]
            for lang, exts in EXTENSIONS.items():
                if ext in exts:
                    f = {lang}
                    break
        else:
            # not a file, so it's an option
            f = OPT2FLAGS.get(opt, f)
        # add all the new flags to the ones we already accumulated
        r |= f
    return r

def query(qstr, tcs):
    # parse query
    pos = set()
    neg = set()
    if len(qstr) > 0:
        for spec in qstr.split(' '):
            if spec[0] == '-':
                neg.add(spec[1:])
            else:
                pos.add(spec[1:] if spec[0] == '+' else spec)

    # process TCs
    def matches(tc):
        flgs = flags(tc)
        return pos.issubset(flgs) and neg.isdisjoint(flgs)

    return list(filter(matches, tcs))

def usage():
    print('usage: %s [-OPTS] [REG_PATH [REG_PATH [...]]]' %
          (sys.argv[0] if len(sys.argv) > 0 else 'regdb.py'))
    print('''
Options:
  -h          display this help message
  -l          list known flags
  -q QUERY    output only tests matching QUERY, multiple queries can be
              performed and each will further restrict the list of results
  -v          increase verbosity (display flags for each test case or
              description for each flag), use twice to show shell-escaped
              cmdline options instead of flags

Multiple paths can optionally be specified as REG_PATH parameters, which will
restrict the test cases to process to only those given. If none are specified,
the default is to search the built-in list of all non-disabled regression tests.

QUERY is a space-separated list of flag specifiers of the form:
  [+]flag : match only TCs that have this flag
    -flag : match only TCs that do not have this flag
All flag specifiers are conjunctively connected.
'''[:-1])
    sys.exit(0)

def main():
    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'hlq:v')
    except getopt.GetoptError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    if ('-h', '') in opts:
        usage()

    if ('-l', '') in opts:
        list_flags(('-v', '') in opts)

    tcs = []
    if args:
        for arg in args:
            tcs.append(TestCase(arg, os.path.basename(arg)))
    else:
        apply_transform_over_tests(tcs.append)

    verbosity = 0
    for opt, val in opts:
        if opt == '-q':
            tcs = query(val, tcs)
        elif opt == '-v':
            verbosity += 1

    if len(tcs) == 0 or verbosity == 0:
        # just directory
        for tc in tcs:
            print(tc.test_dir)
    elif verbosity == 1:
        # directory + flags
        max_w = max(len(tc.test_dir) for tc in tcs)
        for tc in tcs:
            print('%*s: %s' % (-max_w, tc.test_dir, ' '.join(sorted(flags(tc)))))
    else:
        # directory + cmdline options
        max_w = max(len(tc.test_dir) for tc in tcs)
        for tc in tcs:
            print('%*s: %s' % (-max_w, tc.test_dir,
                               shlex.join(tc.generate_run_argument_list('true')[1:])))

if __name__ == '__main__':
    main()
