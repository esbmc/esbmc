"""Programs for the exception census tests: (name, family, census passes, files).

Every program in `reject` form either raises one of the family's exceptions
under CPython 3.14 or uses a construct the census declines on purpose
(reflection, imports it cannot follow, OverflowError-class sites).  The
`accept` programs are the other side of each condition.  The `rev_` programs
come from an independent attack on the census and `sa_` from a second pass: most raise
the family under CPython 3.14, the rest are rejected because they probe a mechanism of the census.  The `atk2_` programs
come from a second independent attack; each raises the family under CPython 3.14.  The `atk3_` programs
come from a third independent attack, with the same property.
"""

CASES = [
    ('a_int_inf', 'arithmetic', False, {
        'main.py': "x = int(float('inf'))\n"
    }),
    ('a_big_pow_mul', 'arithmetic', False, {
        'main.py': 'x = 10**400 * 1.0\n'
    }),
    ('a_float_big', 'arithmetic', False, {
        'main.py': 'x = float(10**400)\n'
    }),
    ('a_round_inf', 'arithmetic', False, {
        'main.py': "x = round(float('inf'))\n"
    }),
    ('a_list_repeat_2_64', 'arithmetic', False, {
        'main.py': 'x = [0] * 2**64\n'
    }),
    ('a_math_exp', 'arithmetic', False, {
        'main.py': 'import math\nx = math.exp(1000)\n'
    }),
    ('a_pow_builtin', 'arithmetic', False, {
        'main.py': 'x = pow(0, -1)\n'
    }),
    ('a_divmod', 'arithmetic', False, {
        'main.py': 'zero = len(())\nx = divmod(1, zero)\n'
    }),
    ('a_mod_number', 'arithmetic', False, {
        'main.py': 'zero = len(())\nx = 7 % zero\n'
    }),
    ('a_mod_float', 'arithmetic', False, {
        'main.py': 'z = 0.0\nx = 7.5 % z\n'
    }),
    ('a_mod_string_number', 'arithmetic', False, {
        'main.py': "zero = len(())\nx = len('ab') % zero\n"
    }),
    ('a_user_add_raises', 'arithmetic', False, {
        'main.py': 'class V:\n    def __add__(self, o):\n        return 1 / 0\nx = V() + V()\n'
    }),
    ('a_user_truediv_raises', 'arithmetic', False, {
        'main.py':
        'class V:\n'
        '    def __truediv__(self, o):\n'
        "        raise ZeroDivisionError('x')\n"
        'x = V() / 3\n'
    }),
    ('a_new_returns_zde', 'arithmetic', False, {
        'main.py':
        'class E:\n'
        '    def __new__(cls):\n'
        "        return ZeroDivisionError('x')\n"
        'raise E()\n'
    }),
    ('a_raise_arith_error', 'arithmetic', False, {
        'main.py': "raise ArithmeticError('x')\n"
    }),
    ('a_raise_zde', 'arithmetic', False, {
        'main.py': "raise ZeroDivisionError('x')\n"
    }),
    ('a_raise_fpe', 'arithmetic', False, {
        'main.py': "raise FloatingPointError('x')\n"
    }),
    ('a_reraise_stored', 'arithmetic', False, {
        'main.py':
        'saved = None\n'
        'try:\n'
        '    1 / len(())\n'
        'except ZeroDivisionError as e:\n'
        '    saved = e\n'
        'raise saved\n'
    }),
    ('a_generator_throw', 'arithmetic', False, {
        'main.py':
        'def g():\n'
        '    yield 1\n'
        '    yield 2\n'
        'it = g()\n'
        'next(it)\n'
        'it.throw(ZeroDivisionError)\n'
    }),
    ('a_truediv_var', 'arithmetic', False, {
        'main.py': 'zero = len(())\nx = 1 / zero\n'
    }),
    ('a_floordiv_var', 'arithmetic', False, {
        'main.py': 'zero = len(())\nx = 1 // zero\n'
    }),
    ('a_augassign_div', 'arithmetic', False, {
        'main.py': 'zero = len(())\nx = 4\nx /= zero\n'
    }),
    ('a_mean_empty', 'arithmetic', False, {
        'main.py': 'xs = []\nm = sum(xs) / len(xs)\n'
    }),
    ('a_pow_neg_const', 'arithmetic', False, {
        'main.py': 'x = 0 ** -1\n'
    }),
    ('a_pow_float_neg', 'arithmetic', False, {
        'main.py': 'x = 0.0 ** -1\n'
    }),
    ('a_pow_var_exp', 'arithmetic', False, {
        'main.py': 'zero = len(())\nx = 0 ** (zero - 1)\n'
    }),
    ('a_math_log_base1', 'arithmetic', False, {
        'main.py': 'import math\nx = math.log(2, 1)\n'
    }),
    ('a_random_expovariate', 'arithmetic', False, {
        'main.py': 'import random\nx = random.expovariate(0)\n'
    }),
    ('a_fraction_zero', 'arithmetic', False, {
        'main.py': 'from fractions import Fraction\nx = Fraction(1, 0)\n'
    }),
    ('a_decimal_zero', 'arithmetic', False, {
        'main.py': 'from decimal import Decimal\nx = Decimal(1) / Decimal(0)\n'
    }),
    ('a_timedelta_zero', 'arithmetic', False, {
        'main.py': 'import datetime\nx = datetime.timedelta(1) / 0\n'
    }),
    ('a_numpy_fpe', 'arithmetic', False, {
        'main.py': "import numpy as np\nnp.seterr(all='raise')\nx = np.float64(1) / np.float64(0)\n"
    }),
    ('a_statistics_import', 'arithmetic', False, {
        'main.py': 'import statistics\nx = statistics.mean([1])\n'
    }),
    ('a_operator_truediv', 'arithmetic', False, {
        'main.py': 'import operator\nx = operator.truediv(1, len(()))\n'
    }),
    ('a_dunder_truediv_call', 'arithmetic', False, {
        'main.py': 'x = int.__truediv__(1, 0)\n'
    }),
    ('a_method_dunder_floordiv', 'arithmetic', False, {
        'main.py': 'zero = len(())\nx = (5).__floordiv__(zero)\n'
    }),
    ('a_lambda_div', 'arithmetic', False, {
        'main.py': 'zero = len(())\nf = lambda a, b: a / b\nx = f(1, zero)\n'
    }),
    ('a_decorator_raises', 'arithmetic', False, {
        'main.py':
        'def deco(f):\n'
        '    def w(*a):\n'
        '        return 1 / 0\n'
        '    return w\n'
        '@deco\n'
        'def g():\n'
        '    return 1\n'
        'g()\n'
    }),
    ('a_metaclass_call', 'arithmetic', False, {
        'main.py':
        'class M(type):\n'
        '    def __call__(cls, *a):\n'
        '        return 1 / 0\n'
        'class C(metaclass=M):\n'
        '    pass\n'
        'C()\n'
    }),
    ('a_getattr_hook', 'arithmetic', False, {
        'main.py': 'class C:\n    def __getattr__(self, n):\n        return 1 / 0\nC().foo\n'
    }),
    ('a_property_div', 'arithmetic', False, {
        'main.py': 'class C:\n    @property\n    def p(self):\n        return 1 / 0\nC().p\n'
    }),
    ('a_rebound_len', 'arithmetic', False, {
        'main.py': 'len = lambda x: 1 / 0\nx = len([])\n'
    }),
    ('a_rebound_exception_name', 'arithmetic', False, {
        'main.py':
        'ZeroDivisionError = ValueError\n'
        'try:\n'
        '    1 / len(())\n'
        'except ZeroDivisionError:\n'
        '    pass\n'
    }),
    ('a_rebound_max', 'arithmetic', False, {
        'main.py': 'max = 64\nx = max\ny = 1 / (max - 64)\n'
    }),
    ('a_except_clause_trick', 'arithmetic', False, {
        'main.py': 'try:\n    x = 1 / len(())\nexcept (ValueError, TypeError):\n    pass\n'
    }),
    ('a_class_attr_class_assign', 'arithmetic', False, {
        'main.py':
        'class A:\n'
        '    pass\n'
        'class B:\n'
        '    def __truediv__(self, o):\n'
        '        return 1 / 0\n'
        'a = A()\n'
        'a.__class__ = B\n'
        'a / 1\n'
    }),
    ('a_eval', 'arithmetic', False, {
        'main.py': "x = eval('1/0')\n"
    }),
    ('a_exec', 'arithmetic', False, {
        'main.py': "exec('x = 1/0')\n"
    }),
    ('a_compile', 'arithmetic', False, {
        'main.py': "c = compile('1/0', 's', 'eval')\nx = eval(c)\n"
    }),
    ('a_globals', 'arithmetic', False, {
        'main.py': "g = globals()\ng['__builtins__']\nx = g['x'] if 'x' in g else 0\n"
    }),
    ('a_vars', 'arithmetic', False, {
        'main.py': 'x = vars()\n'
    }),
    ('a_dunder_import', 'arithmetic', False, {
        'main.py': "x = __import__('math').exp(1000)\n"
    }),
    ('a_importlib', 'arithmetic', False, {
        'main.py': "import importlib\nm = importlib.import_module('math')\nx = m.exp(1000)\n"
    }),
    ('a_getattr_builtin', 'arithmetic', False, {
        'main.py':
        "f = getattr(__builtins__, 'divmod') if hasattr(__builtins__, 'divmod') else "
        'None\n'
    }),
    ('a_dunder_builtins_store', 'arithmetic', False, {
        'main.py': "__builtins__ = {'len': lambda x: 1 / 0}\nx = len([])\n"
    }),
    ('a_shadow_math', 'arithmetic', False, {
        'main.py': 'import math\nx = math.sqrt(4)\n',
        'math.py': 'def sqrt(x):\n    return 1 / 0\n'
    }),
    ('a_shadow_operator_indirect', 'arithmetic', False, {
        'main.py': 'import typing\nx = 1\n',
        'operator.py': 'x = 1 / 0\n'
    }),
    ('a_shadow_string_via_sv', 'arithmetic', False, {
        'main.py': 'import _sv_verifier\nx = 1\n',
        'string.py': 'x = 1 / 0\n'
    }),
    ('a_sibling_import_div', 'arithmetic', False, {
        'helper.py': 'def f():\n    return 1 / len(())\n',
        'main.py': 'import helper\nx = helper.f()\n'
    }),
    ('a_sibling_import_toplevel', 'arithmetic', False, {
        'helper.py': 'x = 1 / 0\n',
        'main.py': 'import helper\n'
    }),
    ('a_from_sibling', 'arithmetic', False, {
        'helper.py': 'def f():\n    return 1 / len(())\n',
        'main.py': 'from helper import f\nf()\n'
    }),
    ('a_relative_import', 'arithmetic', False, {
        'helper.py': 'x = 1 / 0\n',
        'main.py': 'from . import helper\n'
    }),
    ('a_package_init', 'arithmetic', False, {
        'main.py': 'import pkg.sub\n',
        'pkg/__init__.py': 'x = 1 / 0\n',
        'pkg/sub.py': 'y = 1\n'
    }),
    ('a_star_import_rebinding', 'arithmetic', False, {
        'helper.py': 'len = lambda x: 1 / 0\n',
        'main.py': 'from helper import *\nx = len([])\n'
    }),
    ('a_star_dynamic_all', 'arithmetic', False, {
        'helper.py': "__all__ = ['f'] + []\ndef f():\n    return 1 / len(())\n",
        'main.py': 'from helper import *\nx = f()\n'
    }),
    ('a_with_exit_raises', 'arithmetic', False, {
        'main.py':
        'class C:\n'
        '    def __enter__(self):\n'
        '        return self\n'
        '    def __exit__(self, *a):\n'
        '        return 1 / 0\n'
        'with C():\n'
        '    pass\n'
    }),
    ('a_finally_div', 'arithmetic', False, {
        'main.py': 'try:\n    pass\nfinally:\n    x = 1 / len(())\n'
    }),
    ('a_comprehension_div', 'arithmetic', False, {
        'main.py': 'x = [1 / v for v in [0]]\n'
    }),
    ('a_default_arg_div', 'arithmetic', False, {
        'main.py': 'def f(a=1 / 0):\n    pass\n'
    }),
    ('a_class_body_div', 'arithmetic', False, {
        'main.py': 'class C:\n    x = 1 / 0\n'
    }),
    ('a_fstring_div', 'arithmetic', False, {
        'main.py': "x = f'{1 / len(())}'\n"
    }),
    ('a_walrus_div', 'arithmetic', False, {
        'main.py': 'if (y := 1 / len(())):\n    pass\n'
    }),
    ('a_init_subclass', 'arithmetic', False, {
        'main.py':
        'class B:\n'
        '    def __init_subclass__(cls):\n'
        '        1 / 0\n'
        'class C(B):\n'
        '    pass\n'
    }),
    ('a_set_name', 'arithmetic', False, {
        'main.py':
        'class D:\n'
        '    def __set_name__(self, o, n):\n'
        '        1 / 0\n'
        'class C:\n'
        '    d = D()\n'
    }),
    ('a_match_case', 'arithmetic', False, {
        'main.py': 'm = len(())\nmatch m:\n    case 0:\n        x = 1 / m\n'
    }),
    ('a_local_import_in_function', 'arithmetic', False, {
        'main.py': 'def f():\n    import math\n    return math.exp(1000)\nf()\n'
    }),
    ('a_yield_div', 'arithmetic', False, {
        'main.py': 'def g():\n    yield 1 / len(())\nnext(g())\n'
    }),
    ('a_global_pow_rebind', 'arithmetic', False, {
        'main.py': 'def f():\n    global pow\n    return pow(0, -1)\nf()\n'
    }),
    ('a_star_args_pow', 'arithmetic', False, {
        'main.py': 'args = (0, -1)\nx = pow(*args)\n'
    }),
    ('a_map_pow', 'arithmetic', False, {
        'main.py': 'x = list(map(pow, [0], [-1]))\n'
    }),
    ('a_reduce_truediv', 'arithmetic', False, {
        'main.py': 'import functools, operator\nx = functools.reduce(operator.truediv, [1, 0])\n'
    }),
    ('a_type_three_arg', 'arithmetic', False, {
        'main.py': "T = type('T', (), {'__truediv__': lambda s, o: 1 / 0})\nT() / 1\n"
    }),
    ('a_unparsable', 'arithmetic', False, {
        'main.py': 'def f(:\n'
    }),
    ('a_ctl_literal_divisor', 'arithmetic', True, {
        'main.py': "x = len('abcd') / 2\ny = 9 // 4\nz = 7 % 3\nw = 2 ** 10\n"
    }),
    ('a_ctl_const_name_divisor', 'arithmetic', True, {
        'main.py': 'N = 10\nxs = [3, 4]\ny = sum(xs) / N\n'
    }),
    ('a_ctl_string_format', 'arithmetic', True, {
        'main.py': "s = '%d items' % len([1])\n"
    }),
    ('a_ctl_sibling_clean', 'arithmetic', True, {
        'helper.py': 'def f(a):\n    return a + 1\n',
        'main.py': 'import helper\nx = helper.f(3)\n'
    }),
    ('a_ctl_class_methods', 'arithmetic', True, {
        'main.py':
        'class P:\n'
        '    def __init__(self, x):\n'
        '        self.x = x\n'
        '    def get(self):\n'
        '        return self.x\n'
        'p = P(3)\n'
        'y = p.get() + 1\n'
    }),
    ('a_ctl_math_sqrt', 'arithmetic', True, {
        'main.py': 'import math\nx = math.sqrt(16.0) + math.pi\n'
    }),
    ('a_ctl_loop_list', 'arithmetic', True, {
        'main.py': 'xs = []\nfor i in range(5):\n    xs.append(i * 2)\ns = sum(xs)\n'
    }),
    ('a_ctl_data_ops', 'arithmetic', True, {
        'main.py': "d = {}\nd['a'] = 1\nv = d['a']\nxs = [1]\nxs.pop()\nxs.pop() if xs else 0\n"
    }),
    ('a_ctl_typing', 'arithmetic', True, {
        'main.py':
        'from typing import List, Dict\n'
        'def f(x: List[int], y: Dict[str, int]) -> int:\n'
        '    return len(x)\n'
        'f([1], {})\n'
    }),
    ('a_ctl_plain_function', 'arithmetic', True, {
        'main.py': 'def f(x):\n    return x + 1\nf(2)\n'
    }),
    ('d_dict_missing', 'data-lookup', False, {
        'main.py': 'd = {}\nx = d[1]\n'
    }),
    ('d_list_oob', 'data-lookup', False, {
        'main.py': 'xs = []\nx = xs[0]\n'
    }),
    ('d_slice_of_dict_3_12', 'data-lookup', False, {
        'main.py': 'd = {}\nx = d[1:2]\n'
    }),
    ('d_slice_getitem_raises', 'data-lookup', False, {
        'main.py':
        'class C:\n'
        '    def __getitem__(self, i):\n'
        "        raise IndexError('x')\n"
        'x = C()[1:2]\n'
    }),
    ('d_list_pop_empty', 'data-lookup', False, {
        'main.py': 'xs = []\nxs.pop()\n'
    }),
    ('d_list_pop_zero', 'data-lookup', False, {
        'main.py': 'xs = []\nxs.pop(0)\n'
    }),
    ('d_dict_pop_missing', 'data-lookup', False, {
        'main.py': "d = {}\nd.pop('a')\n"
    }),
    ('d_set_pop_empty', 'data-lookup', False, {
        'main.py': 's = set()\ns.pop()\n'
    }),
    ('d_set_remove', 'data-lookup', False, {
        'main.py': 's = set()\ns.remove(1)\n'
    }),
    ('d_popitem_empty', 'data-lookup', False, {
        'main.py': 'd = {}\nd.popitem()\n'
    }),
    ('d_deque_popleft', 'data-lookup', False, {
        'main.py': 'from collections import deque\nd = deque()\nd.popleft()\n'
    }),
    ('d_deque_pop', 'data-lookup', False, {
        'main.py': 'from collections import deque\ndeque().pop()\n'
    }),
    ('d_heappop', 'data-lookup', False, {
        'main.py': 'from heapq import heappop\nheappop([])\n'
    }),
    ('d_heappop_attr', 'data-lookup', False, {
        'main.py': 'import heapq\nheapq.heappop([])\n'
    }),
    ('d_heapreplace', 'data-lookup', False, {
        'main.py': 'import heapq\nheapq.heapreplace([], 1)\n'
    }),
    ('d_random_choice', 'data-lookup', False, {
        'main.py': 'import random\nrandom.choice([])\n'
    }),
    ('d_random_choices', 'data-lookup', False, {
        'main.py': 'import random\nrandom.choices([], k=1)\n'
    }),
    ('d_format_index', 'data-lookup', False, {
        'main.py': "x = '{0}'.format()\n"
    }),
    ('d_format_key', 'data-lookup', False, {
        'main.py': "x = '{a}'.format()\n"
    }),
    ('d_format_map', 'data-lookup', False, {
        'main.py': "x = '{a}'.format_map({})\n"
    }),
    ('d_re_group', 'data-lookup', False, {
        'main.py': "import re\nm = re.match('a', 'a')\nm.group(5)\n"
    }),
    ('d_itemgetter', 'data-lookup', False, {
        'main.py': 'import operator\noperator.itemgetter(5)([1])\n'
    }),
    ('d_sys_argv', 'data-lookup', False, {
        'main.py': 'import sys\nx = sys.argv[1]\n'
    }),
    ('d_functools_reduce_getitem', 'data-lookup', False, {
        'main.py': 'import functools\nfunctools.reduce(lambda a, b: a[b], [5], [1])\n'
    }),
    ('d_next_raising_generator', 'data-lookup', False, {
        'main.py': 'def g():\n    yield [][0]\nnext(g())\n'
    }),
    ('d_contains_dunder', 'data-lookup', False, {
        'main.py':
        'class C:\n'
        '    def __contains__(self, x):\n'
        "        raise IndexError('x')\n"
        '1 in C()\n'
    }),
    ('d_iter_dunder', 'data-lookup', False, {
        'main.py':
        'class C:\n'
        '    def __iter__(self):\n'
        "        raise KeyError('x')\n"
        'for _ in C():\n'
        '    pass\n'
    }),
    ('d_with_dunder', 'data-lookup', False, {
        'main.py':
        'class C:\n'
        '    def __enter__(self):\n'
        "        raise KeyError('x')\n"
        '    def __exit__(self, *a):\n'
        '        pass\n'
        'with C():\n'
        '    pass\n'
    }),
    ('d_str_dunder', 'data-lookup', False, {
        'main.py': "class C:\n    def __str__(self):\n        raise KeyError('x')\nstr(C())\n"
    }),
    ('d_bool_dunder', 'data-lookup', False, {
        'main.py': "class C:\n    def __bool__(self):\n        raise IndexError('x')\nbool(C())\n"
    }),
    ('d_len_dunder', 'data-lookup', False, {
        'main.py': 'class C:\n    def __len__(self):\n        return [][0]\nlen(C())\n'
    }),
    ('d_fstring_format_dunder', 'data-lookup', False, {
        'main.py':
        'class C:\n'
        '    def __format__(self, spec):\n'
        "        raise KeyError('x')\n"
        "f'{C()}'\n"
    }),
    ('d_compare_dunder', 'data-lookup', False, {
        'main.py': "class C:\n    def __lt__(self, o):\n        raise IndexError('x')\nC() < C()\n"
    }),
    ('d_percent_dict', 'data-lookup', False, {
        'main.py': "x = '%(k)s' % {}\n"
    }),
    ('d_percent_name', 'data-lookup', False, {
        'main.py': "fmt = '%(k)s'\nx = fmt % {}\n"
    }),
    ('d_getattr_builtin', 'data-lookup', False, {
        'main.py': "getattr([], 'pop')()\n"
    }),
    ('d_dunder_getitem_call', 'data-lookup', False, {
        'main.py': 'd = {}\nd.__getitem__(1)\n'
    }),
    ('d_sorted_key_getitem', 'data-lookup', False, {
        'main.py': 'd = {}\nsorted([1], key=d.__getitem__)\n'
    }),
    ('d_namedtuple_functional', 'data-lookup', False, {
        'main.py': 'from typing import NamedTuple\nNamedTuple({})\n'
    }),
    ('d_ordereddict_move_to_end', 'data-lookup', False, {
        'main.py': "from collections import OrderedDict\nOrderedDict().move_to_end('a')\n"
    }),
    ('d_counter_pop', 'data-lookup', False, {
        'main.py': "from collections import Counter\nCounter().pop('a')\n"
    }),
    ('d_array_pop', 'data-lookup', False, {
        'main.py': "import array\narray.array('i').pop()\n"
    }),
    ('d_bytearray_pop', 'data-lookup', False, {
        'main.py': 'bytearray().pop()\n'
    }),
    ('d_random_setstate', 'data-lookup', False, {
        'main.py': "import random\nrandom.setstate('')\n"
    }),
    ('d_del_subscript', 'data-lookup', False, {
        'main.py': 'xs = []\ndel xs[0]\n'
    }),
    ('d_store_subscript', 'data-lookup', False, {
        'main.py': 'xs = []\nxs[0] = 1\n'
    }),
    ('d_augassign_subscript', 'data-lookup', False, {
        'main.py': "d = {}\nd['a'] += 1\n"
    }),
    ('d_enum_getitem', 'data-lookup', False, {
        'main.py': "import enum\nclass E(enum.Enum):\n    A = 1\nE['B']\n"
    }),
    ('d_globals_subscript', 'data-lookup', False, {
        'main.py': "globals()['nope']\n"
    }),
    ('d_os_environ', 'data-lookup', False, {
        'main.py': "import os\nos.environ['NOPE_X']\n"
    }),
    ('d_template_substitute', 'data-lookup', False, {
        'main.py': "import string\nstring.Template('$a').substitute({})\n"
    }),
    ('d_locals_subscript', 'data-lookup', False, {
        'main.py': "def f():\n    return locals()['nope']\nf()\n"
    }),
    ('d_builtins_subscript', 'data-lookup', False, {
        'main.py':
        "__builtins__['nope'] if isinstance(__builtins__, dict) else "
        "__builtins__.__dict__['nope']\n"
    }),
    ('d_shadow_heapq', 'data-lookup', False, {
        'heapq.py': 'x = [][0]\n',
        'main.py': 'import heapq\nx = 1\n'
    }),
    ('d_sibling_subscript', 'data-lookup', False, {
        'helper.py': 'def f():\n    return [][0]\n',
        'main.py': 'import helper\nhelper.f()\n'
    }),
    ('d_rebound_list', 'data-lookup', False, {
        'main.py': 'list = {}\nx = list[1]\n'
    }),
    ('d_type_subscript_rebound', 'data-lookup', False, {
        'main.py': 'from typing import List\nList = {}\nx = List[1]\n'
    }),
    ('d_type_subscript_param', 'data-lookup', False, {
        'main.py': 'def f(dict):\n    return dict[1]\nf({})\n'
    }),
    ('d_annotation_subscript', 'data-lookup', False, {
        'main.py': 'd = {}\ndef f(x: d[1]):\n    pass\n'
    }),
    ('d_percent_tuple_short', 'data-lookup', False, {
        'main.py': "x = '%s %s' % (1,)\n"
    }),
    ('d_unparsable', 'data-lookup', False, {
        'main.py': 'class (:\n'
    }),
    ('d_ctl_get_and_loops', 'data-lookup', True, {
        'main.py': "d = {'a': 1}\nv = d.get('b', 0)\nfor k in d:\n    v += 1\n"
    }),
    ('d_ctl_typed_signature', 'data-lookup', True, {
        'main.py':
        'def f(x: list[int], y: dict[str, int]) -> tuple[int, ...]:\n'
        '    return tuple(x)\n'
        'f([1], {})\n'
    }),
    ('d_ctl_typing_generics', 'data-lookup', True, {
        'main.py':
        'from typing import List, Optional\n'
        'def f(x: Optional[List[int]] = None) -> int:\n'
        '    return 0 if x is None else len(x)\n'
        'f()\n'
    }),
    ('d_ctl_numeric_percent', 'data-lookup', True, {
        'main.py': "n = len('abc')\nx = n % 2\ny = 5 % n\n"
    }),
    ('d_ctl_sets_lists', 'data-lookup', True, {
        'main.py':
        's = set()\n'
        's.add(1)\n'
        'xs = sorted(s)\n'
        'xs.append(2)\n'
        'xs.sort()\n'
        "z = ','.join(str(i) for i in xs)\n"
    }),
    ('d_ctl_zero_div_outside_family', 'data-lookup', True, {
        'main.py': 'x = 1 / len(())\n'
    }),
    ('d_ctl_class_dataclass', 'data-lookup', True, {
        'main.py':
        'from dataclasses import dataclass\n'
        '@dataclass\n'
        'class P:\n'
        '    x: int\n'
        '    y: int = 0\n'
        'p = P(1)\n'
        'q = p.x + p.y\n'
    }),
    ('d_ctl_enum', 'data-lookup', False, {
        'main.py': 'import enum\nclass E(enum.Enum):\n    A = 1\nx = E.A.value\n'
    }),
    ('d_ctl_sibling_clean', 'data-lookup', True, {
        'helper.py': 'def f(a):\n    return a + 1\n',
        'main.py': 'import helper\nx = helper.f(3)\n'
    }),
    ('d_ctl_in_operator', 'data-lookup', True, {
        'main.py': "d = {}\nb = 'k' in d\nc = d.items()\nm = min([1, 2], default=0)\n"
    }),
    ('d_alias_rebound', 'data-lookup', False, {
        'helper.py': 'pop = 1\n',
        'main.py': 'import helper\nhelper = []\nhelper.pop()\n'
    }),
    ('d_alias_module_attr_safe', 'data-lookup', True, {
        'helper.py': 'pop = 1\n',
        'main.py': 'import helper\nx = helper.pop\n'
    }),
    ('d_dict_subclass_pop', 'data-lookup', False, {
        'main.py': "class D(dict):\n    pass\nD().pop('a')\n"
    }),
    ('d_super_getitem', 'data-lookup', False, {
        'main.py':
        'class D(dict):\n'
        '    def g(self, k):\n'
        '        return super().__getitem__(k)\n'
        'D().g(1)\n'
    }),
    ('d_user_method_named_pop_on_list', 'data-lookup', False, {
        'main.py': 'class S:\n    def pop(self):\n        return 1\nxs = []\nxs.pop()\n'
    }),
    ('a_const_name_global_rebound', 'arithmetic', False, {
        'main.py': 'N = 10\ndef f():\n    global N\n    N = 0\nf()\nx = 5 / N\n'
    }),
    ('a_const_name_augassign', 'arithmetic', False, {
        'main.py': 'N = 1\nN -= 1\nx = 5 / N\n'
    }),
    ('a_const_name_loop_var', 'arithmetic', False, {
        'main.py': 'N = 1\nfor N in [0]:\n    pass\nx = 5 / N\n'
    }),
    ('a_const_name_star_import', 'arithmetic', False, {
        'helper.py': 'N = 0\n',
        'main.py': 'N = 1\nfrom helper import *\nx = 5 / N\n'
    }),
    ('a_const_name_class_shadow', 'arithmetic', False, {
        'main.py': 'N = 1\nclass C:\n    N = 0\n    x = 5 / N\n'
    }),
    ('a_const_name_param_shadow', 'arithmetic', False, {
        'main.py': 'N = 1\ndef f(N):\n    return 5 / N\nf(0)\n'
    }),
    ('a_const_name_attr_store', 'arithmetic', False, {
        'helper.py': 'N = 1\n',
        'main.py': 'import helper\nN = 1\nhelper.N = 0\nx = 5 / helper.N\n'
    }),
    ('a_const_name_ctl', 'arithmetic', True, {
        'main.py': 'N = 10\ndef f(a):\n    return a / N\nx = f(3)\n'
    }),
    ('a_try_import_fallback', 'arithmetic', False, {
        'main.py':
        'try:\n'
        '    import numpy as np\n'
        'except ImportError:\n'
        '    np = None\n'
        'x = 1 / len(())\n'
    }),
    ('a_dataclass_post_init', 'arithmetic', False, {
        'main.py':
        'from dataclasses import dataclass\n'
        '@dataclass\n'
        'class P:\n'
        '    x: int\n'
        '    def __post_init__(self):\n'
        '        1 / 0\n'
        'P(1)\n'
    }),
    ('a_walrus_comprehension', 'arithmetic', False, {
        'main.py': 'xs = [y := 1 / len(()) for _ in [0]]\n'
    }),
    ('a_pow_neg_const_base_var', 'arithmetic', False, {
        'main.py': 'z = len(())\nx = z ** -1\n'
    }),
    ('a_float_mod_zero_literal', 'arithmetic', False, {
        'main.py': 'x = 5.0 % 0.0\n'
    }),
    ('a_floordiv_zero_literal', 'arithmetic', False, {
        'main.py': 'x = 5 // 0\n'
    }),
    ('a_div_complex_zero', 'arithmetic', False, {
        'main.py': 'x = 1 / 0j\n'
    }),
    ('a_bool_false_divisor', 'arithmetic', False, {
        'main.py': 'x = 1 / False\n'
    }),
    ('a_sum_len_guard', 'arithmetic', False, {
        'main.py': 'xs = []\nx = sum(xs) / len(xs) if len(xs) > 1 else sum(xs) / len(xs)\n'
    }),
    ('a_lambda_in_default', 'arithmetic', False, {
        'main.py': 'def f(g=lambda: 1 / 0):\n    return g()\nf()\n'
    }),
    ('a_class_decorator', 'arithmetic', False, {
        'main.py': 'def deco(c):\n    1 / 0\n    return c\n@deco\nclass C:\n    pass\n'
    }),
    ('a_property_deleter', 'arithmetic', False, {
        'main.py':
        'class C:\n'
        '    @property\n'
        '    def p(self):\n'
        '        return 1\n'
        '    @p.deleter\n'
        '    def p(self):\n'
        '        1 / 0\n'
        'c = C()\n'
        'del c.p\n'
    }),
    ('a_iter_next_div', 'arithmetic', False, {
        'main.py':
        'class I:\n'
        '    def __iter__(self):\n'
        '        return self\n'
        '    def __next__(self):\n'
        '        return 1 / 0\n'
        'for _ in I():\n'
        '    pass\n'
    }),
    ('a_hash_div', 'arithmetic', False, {
        'main.py': 'class H:\n    def __hash__(self):\n        return 1 // 0\n{H(): 1}\n'
    }),
    ('a_eq_div', 'arithmetic', False, {
        'main.py': 'class H:\n    def __eq__(self, o):\n        return 1 / 0\nH() == H()\n'
    }),
    ('a_index_dunder_div', 'arithmetic', False, {
        'main.py': 'class H:\n    def __index__(self):\n        return 1 // 0\n[1][H()]\n'
    }),
    ('a_del_dunder', 'arithmetic', False, {
        'main.py': 'class H:\n    def __del__(self):\n        1 / 0\nH()\n'
    }),
    ('a_nested_class_scope_pow', 'arithmetic', False, {
        'main.py':
        'class C:\n'
        '    def pow(self, a, b):\n'
        '        return a\n'
        '    def f(self):\n'
        '        return pow(0, -1)\n'
        'C().f()\n'
    }),
    ('a_comp_target_shadow', 'arithmetic', False, {
        'main.py': 'x = [pow(0, -1) for pow2 in [1]]\n'
    }),
    ('d_class_scope_list_shadow', 'data-lookup', False, {
        'main.py': 'class C:\n    list = {}\n    x = list[1]\n'
    }),
    ('d_nested_function_dict', 'data-lookup', False, {
        'main.py': 'def f():\n    d = {}\n    def g():\n        return d[1]\n    return g()\nf()\n'
    }),
    ('d_generator_expression_index', 'data-lookup', False, {
        'main.py': 'xs = []\nvals = list(xs[i] for i in range(1))\n'
    }),
    ('d_lambda_index', 'data-lookup', False, {
        'main.py': 'f = lambda xs: xs[0]\nf([])\n'
    }),
    ('d_default_index', 'data-lookup', False, {
        'main.py': 'def f(a=[][0]):\n    pass\n'
    }),
    ('d_index_in_fstring', 'data-lookup', False, {
        'main.py': "xs = []\ns = f'{xs[0]}'\n"
    }),
    ('d_index_in_decorator', 'data-lookup', False, {
        'main.py': 'def deco(f):\n    [][0]\n    return f\n@deco\ndef g():\n    pass\n'
    }),
    ('d_stdlib_shadow_collections', 'data-lookup', False, {
        'collections.py': 'x = [][0]\n',
        'main.py': 'from collections import deque\nx = 1\n'
    }),
    ('d_overflow_not_family', 'data-lookup', True, {
        'main.py': "x = int(float('inf')) if False else 1\ny = 10**400 * 1.0 if False else 1\n"
    }),
    ('d_deque_insert_full', 'data-lookup', False, {
        'main.py': 'from collections import deque\nd = deque([1], maxlen=1)\nd.insert(0, 2)\n'
    }),
    ('d_random_shuffle_dict', 'data-lookup', False, {
        'main.py': "import random\nrandom.shuffle({'a': 1, 'b': 2})\n"
    }),
    ('d_from_import_shuffle', 'data-lookup', False, {
        'main.py': "from random import shuffle\nshuffle({'a': 1, 'b': 2})\n"
    }),
    ('d_typing_attr_namedtuple', 'data-lookup', False, {
        'main.py': 'import typing\ntyping.NamedTuple({})\n'
    }),
    ('d_typing_attr_typeddict', 'data-lookup', False, {
        'main.py': 'import typing\ntyping.TypedDict({})\n'
    }),
    ('d_list_insert_name', 'data-lookup', False, {
        'main.py': 'xs = []\nxs.insert(0, 1)\n'
    }),
    ('a_shadow_random', 'arithmetic', False, {
        'main.py': 'import random\nx = random.random()\n',
        'random.py': 'x = 1 / 0\n'
    }),
    ('a_sitecustomize', 'arithmetic', False, {
        'main.py': 'x = 1\n',
        'sitecustomize.py': 'y = 1 / 0\n'
    }),
    ('d_shadow_typing', 'data-lookup', False, {
        'main.py': 'import typing\nx = 1\n',
        'typing.py': 'x = [][0]\n'
    }),
    ('rev_a1_pkg_attr_replaced_D', 'data-lookup', False, {
        'main.py': 'import pkg.helper\npkg.helper = {}\npkg.helper.pop("k")\n',
        'pkg/__init__.py': '',
        'pkg/helper.py': 'def pop(k):\n    return k\n'
    }),
    ('rev_a2_init_attr_beats_submodule_D', 'data-lookup', False, {
        'main.py': 'from pkg import helper\nhelper.pop("k")\n',
        'pkg/__init__.py': 'helper = {}\n',
        'pkg/helper.py': 'def pop(k):\n    return k\n'
    }),
    ('rev_b1_import_as_dunder_D', 'data-lookup', False, {
        'h.py':
        'def m(self, k):\n    return k\n',
        'main.py':
        'class D(dict):\n'
        '    from h import m as __missing__\n'
        'print(D()["x"] if False else D().get("x"))\n'
    }),
    ('rev_b2_type_missing_D', 'data-lookup', False, {
        'main.py':
        'D = type("D", (dict,), {"__missing__": lambda self, k: k, "__class_getitem__": '
        'lambda c, k: k})\n'
        'print(D().get("x"))\n'
    }),
    ('rev_i10_deepcopy_list_memo_D', 'data-lookup', False, {
        'main.py':
        'import copy\n'
        'class M(list):\n'
        '    def get(self, k, d=None):\n'
        '        return d\n'
        'print(copy.deepcopy([1], M()))\n'
    }),
    ('rev_i11_enum_functional_mapping_D', 'data-lookup', False, {
        'main.py':
        'from enum import Enum\n'
        'class D(dict):\n'
        '    def __iter__(self):\n'
        '        return iter(["x"])\n'
        'print(Enum("E", D()))\n'
    }),
    ('rev_i12_sample_empty_counts_D', 'data-lookup', False, {
        'main.py': 'import random\nprint(random.sample([], 0, counts=[]))\n'
    }),
    ('rev_i13_normalvariate_A', 'arithmetic', False, {
        'main.py':
        'import random\n'
        'class R(random.Random):\n'
        '    def random(self):\n'
        '        return 1.0\n'
        'print(R().normalvariate(0, 1))\n'
    }),
    ('rev_i1_dict_ctor_keys_D', 'data-lookup', False, {
        'main.py': 'class T(list):\n    def keys(self):\n        return [0]\nprint(dict(T()))\n'
    }),
    ('rev_i2_dict_display_keys_D', 'data-lookup', False, {
        'main.py': 'class T(list):\n    def keys(self):\n        return [0]\nprint({**T()})\n'
    }),
    ('rev_i3_call_star_star_keys_D', 'data-lookup', False, {
        'main.py':
        'class T(list):\n'
        '    def keys(self):\n'
        '        return [0]\n'
        'def f(**k):\n'
        '    return k\n'
        'print(f(**T()))\n'
    }),
    ('rev_i4_update_keys_D', 'data-lookup', False, {
        'main.py':
        'class T(list):\n'
        '    def keys(self):\n'
        '        return [0]\n'
        'd = {}\n'
        'd.update(T())\n'
    }),
    ('rev_i5_counter_items_D', 'data-lookup', False, {
        'main.py':
        'from collections import Counter\n'
        'class C(Counter):\n'
        '    def items(self):\n'
        '        return [(0,)]\n'
        'print(C().most_common())\n'
    }),
    ('rev_i6_heappushpop_lt_clear_D', 'data-lookup', False, {
        'main.py':
        'import heapq\n'
        'h = []\n'
        'class X:\n'
        '    def __lt__(self, o):\n'
        '        h.clear()\n'
        '        return True\n'
        'h.append(X())\n'
        'print(heapq.heappushpop(h, X()))\n'
    }),
    ('rev_i7_odict_dict_clear_D', 'data-lookup', False, {
        'main.py':
        'from collections import OrderedDict\n'
        'od = OrderedDict(a=1)\n'
        'dict.clear(od)\n'
        'print(list(od.items()))\n'
    }),
    ('rev_i8_sample_lying_len_D', 'data-lookup', False, {
        'main.py':
        'import random\n'
        'class L(list):\n'
        '    def __len__(self):\n'
        '        return 1000000\n'
        'print(random.sample(L([1]), 1))\n'
    }),
    ('rev_i9_random_randbelow_D', 'data-lookup', False, {
        'main.py':
        'import random\n'
        'class R(random.Random):\n'
        '    def _randbelow(self, n):\n'
        '        return n + 5\n'
        'print(R().sample([1, 2, 3], 2))\n'
    }),
    ('rev_m1_mapping_values_list_D', 'data-lookup', False, {
        'main.py': 'from collections.abc import Mapping\nprint(list(Mapping.values([5, 6])))\n'
    }),
    ('rev_m2_typing_mapping_values_D', 'data-lookup', False, {
        'main.py': 'from typing import Mapping\nprint(list(Mapping.values([5, 6])))\n'
    }),
    ('rev_m3_sequence_index_dict_D', 'data-lookup', False, {
        'main.py': 'from collections.abc import Sequence\nprint(Sequence.index({}, 1))\n'
    }),
    ('rev_m4_mapping_get_list_D', 'data-lookup', False, {
        'main.py': 'from collections.abc import Mapping\nprint(Mapping.get([], 0))\n'
    }),
    ('rev_n1_ns_fractions_from_A', 'arithmetic', False, {
        'fractions/Fraction.py': 'x = 1\n',
        'main.py': 'from fractions import Fraction\nprint(Fraction(1, 0))\n'
    }),
    ('rev_n2_ns_operator_from_D', 'data-lookup', False, {
        'main.py': 'from operator import getitem\nprint(getitem({}, "k"))\n',
        'operator/getitem.py': 'x = 1\n'
    }),
    ('rev_n3_ns_fractions_import_A', 'arithmetic', False, {
        'fractions/keep.txt': '',
        'main.py': 'import fractions\nclass X:\n    Fraction = 1\nprint(fractions.Fraction(1, 0))\n'
    }),
    ('rev_p1_typing_list_store_D', 'data-lookup', False, {
        'main.py': 'import typing\ntyping.List = []\nprint(typing.List[5])\n'
    }),
    ('rev_p2_string_printable_helper_D', 'data-lookup', False, {
        'main.py':
        'import string\n'
        'from _sv_verifier import nondet_str\n'
        'string.printable = ""\n'
        'for _ in range(1000):\n'
        '    nondet_str()\n'
    }),
    ('rev_p3_random_class_patch_A', 'arithmetic', False, {
        'main.py':
        'import random\n'
        'random.Random.random = lambda self: 1.0\n'
        'print(random.normalvariate(0, 1))\n'
    }),
    ('rev_s1_class_local_builtin_A', 'arithmetic', False, {
        'main.py':
        'def f(divmod=None):\n'
        '    class A:\n'
        '        x = divmod(1, 0)\n'
        '        divmod = None\n'
        'f()\n'
    }),
    ('rev_s1_class_local_builtin_D', 'data-lookup', False, {
        'main.py':
        'def f(eval=None):\n'
        '    class A:\n'
        '        x = eval("{}[\'k\']")\n'
        '        eval = None\n'
        'f()\n'
    }),
    ('rev_s2_class_global_A', 'arithmetic', False, {
        'main.py':
        'def f(divmod=None):\n'
        '    class A:\n'
        '        global divmod\n'
        '        x = divmod(1, 0)\n'
        'f()\n'
    }),
    ('rev_s3_lambda_walrus_in_comp_A', 'arithmetic', False, {
        'main.py':
        'def f():\n'
        '    [(lambda: (divmod := 1)) for _ in ()]\n'
        '    return divmod(1, 0)\n'
        'f()\n'
    }),
    ('rev_s3_lambda_walrus_in_comp_D', 'data-lookup', False, {
        'main.py':
        'def f():\n'
        '    [(lambda: (eval := 1)) for _ in ()]\n'
        '    return eval("{}[\'k\']")\n'
        'f()\n'
    }),
    ('rev_s4_comp_target_load_A', 'arithmetic', False, {
        'main.py': 'a = [0]\nprint([0 for a[divmod(1, 0)] in [1]])\n'
    }),
    ('rev_s4_comp_target_load_D', 'data-lookup', False, {
        'main.py': 'print([0 for eval("{}[\'k\']").x in [1]])\n'
    }),
    ('rev_u1_enum_member_map_D', 'data-lookup', False, {
        'main.py':
        'from enum import Enum\n'
        'class C:\n'
        '    _member_map_ = None\n'
        'class E(Enum):\n'
        '    a = 1\n'
        'E._member_map_.clear()\n'
        'print(list(E))\n'
    }),
    ('rev_v10_metaclass_from_type_D', 'data-lookup', False, {
        'main.py': 'T = type(int)("T", (list,), {"keys": lambda self: [0]})\nprint(dict(T()))\n'
    }),
    ('rev_v11_local_class_named_list_D', 'data-lookup', False, {
        'main.py':
        'def f():\n'
        '    class list:\n'
        '        pass\n'
        'class T(list):\n'
        '    def keys(self):\n'
        '        return [0]\n'
        'print(dict(T()))\n'
    }),
    ('rev_v12_mutableset_clear_list_D', 'data-lookup', False, {
        'main.py': 'from collections.abc import MutableSet\nMutableSet.clear([1])\n'
    }),
    ('rev_v13_mutableseq_clear_set_D', 'data-lookup', False, {
        'main.py': 'from collections.abc import MutableSequence\nMutableSequence.clear({1})\n'
    }),
    ('rev_v14_counter_update_foreign_D', 'data-lookup', False, {
        'main.py': 'from collections import Counter\nCounter.update([], [1])\n'
    }),
    ('rev_v15_counter_subtract_foreign_D', 'data-lookup', False, {
        'main.py': 'from collections import Counter\nCounter.subtract([], [1])\n'
    }),
    ('rev_v1_user_subclass_of_mapping_D', 'data-lookup', False, {
        'main.py':
        'from collections.abc import Mapping\n'
        'class Q(Mapping):\n'
        '    def __getitem__(self, k):\n'
        '        return k\n'
        '    def __iter__(self):\n'
        '        return iter(())\n'
        '    def __len__(self):\n'
        '        return 0\n'
        'print(list(Q.values([5, 6])))\n'
    }),
    ('rev_v2_alias_mapping_D', 'data-lookup', False, {
        'main.py':
        'from collections.abc import Mapping\n'
        'M = Mapping\n'
        'print(list(M.values([5, 6])))\n'
    }),
    ('rev_v3_three_arg_type_D', 'data-lookup', False, {
        'main.py': 'T = type("T", (list,), {"keys": lambda self: [0]})\nprint(dict(T()))\n'
    }),
    ('rev_v4_alias_base_D', 'data-lookup', False, {
        'main.py':
        'B = list\n'
        'class T(B):\n'
        '    def keys(self):\n'
        '        return [0]\n'
        'print(dict(T()))\n'
    }),
    ('rev_v5_alias_store_D', 'data-lookup', False, {
        'main.py':
        'import string\n'
        'from _sv_verifier import nondet_str\n'
        's = string\n'
        's.printable = ""\n'
        'for _ in range(1000):\n'
        '    nondet_str()\n'
    }),
    ('rev_v6_helper_string_store_D', 'data-lookup', False, {
        'main.py':
        'import _sv_verifier as sv\n'
        'class Empty:\n'
        '    printable = ""\n'
        'sv.string = Empty\n'
        'for _ in range(1000):\n'
        '    sv.nondet_str()\n'
    }),
    ('rev_v7_lying_iter_dict_D', 'data-lookup', False, {
        'main.py':
        'class D(dict):\n'
        '    def __iter__(self):\n'
        '        return iter(["x"])\n'
        '    def keys(self):\n'
        '        return ["x"]\n'
        'print(dict(D()))\n'
    }),
    ('rev_v8_mutableseq_reverse_D', 'data-lookup', False, {
        'main.py':
        'from collections.abc import MutableSequence\n'
        'print(MutableSequence.reverse(type("L", (list,), {"__len__": lambda self: '
        '4})()))\n'
    }),
    ('rev_v9_odict_hash_change_D', 'data-lookup', False, {
        'main.py':
        'from collections import OrderedDict\n'
        'class K:\n'
        '    n = 0\n'
        '    def __hash__(self):\n'
        '        K.n += 1\n'
        '        return K.n\n'
        'od = OrderedDict()\n'
        'od.setdefault(K(), 1)\n'
        'print(list(od))\n'
    }),
    ('sa_class_cond_local', 'arithmetic', False, {
        'main.py':
        'flag = len(())\n'
        'class C:\n'
        '    if flag:\n'
        '        divmod = 1\n'
        '    x = divmod(1, 0)\n'
    }),
    ('sa_default_arg_builtin', 'arithmetic', False, {
        'main.py': 'def f(a=divmod(1, len(()))):\n    divmod = 1\nf()\n'
    }),
    ('sa_comp_var_not_leaked', 'arithmetic', False, {
        'main.py': '[1 for pow in []]\nx = pow(0, -1)\n'
    }),
    ('sa_lambda_param_shadow', 'arithmetic', False, {
        'main.py': 'f = lambda divmod: 1\nx = divmod(1, 0)\n'
    }),
    ('sa_except_as_unbound', 'arithmetic', False, {
        'main.py': 'try:\n    pass\nexcept ValueError as divmod:\n    pass\nx = divmod(1, 0)\n'
    }),
    ('sa_class_alias_counter_as', 'data-lookup', False, {
        'main.py': 'from collections import Counter as C\nC.update([], [1])\n'
    }),
    ('sa_class_alias_mapping_as', 'data-lookup', False, {
        'main.py':
        'from collections.abc import Mapping as M\n'
        'class D2(dict):\n'
        '    pass\n'
        'M.values({})\n'
        'M.get([], 0)\n'
    }),
    ('sa_typing_counter_attr', 'data-lookup', False, {
        'main.py': 'import typing\ntyping.Counter.update([], [1])\n'
    }),
    ('sa_typing_mapping_alias', 'data-lookup', False, {
        'main.py': 'from typing import Mapping as M\nM.get([], 0)\n'
    }),
    ('sa_random_class_as', 'data-lookup', False, {
        'main.py': 'from random import Random as R\nR.sample(R(), [], 0, counts=[])\n'
    }),
    ('sa_assign_alias_of_counter', 'data-lookup', False, {
        'main.py': 'from collections import Counter\nc = Counter\nc.update([], [1])\n'
    }),
    ('sa_enum_alias_value', 'data-lookup', False, {
        'main.py': 'from enum import Enum\ne = Enum\nx = e\n'
    }),
    ('sa_module_store_nonuniverse_then_lib', 'arithmetic', False, {
        'main.py': 'import math\nmath.sqrt = lambda x: 1 / len(())\ny = math.sqrt(4)\n'
    }),
    ('sa_augassign_module_attr', 'arithmetic', False, {
        'main.py': 'import math\nmath.pi += 1\n'
    }),
    ('sa_for_target_module_attr', 'arithmetic', False, {
        'main.py':
        'import random\n'
        'for random.randint in [lambda a, b: 1 / len(())]:\n'
        '    pass\n'
        'x = random.randint(1, 2)\n'
    }),
    ('sa_with_target_attr', 'arithmetic', False, {
        'main.py':
        'import math\n'
        'class W:\n'
        '    def __enter__(self):\n'
        '        return lambda x: 1 / len(())\n'
        '    def __exit__(self, *a):\n'
        '        return False\n'
        'with W() as math.sqrt:\n'
        '    pass\n'
        'math.sqrt(1)\n'
    }),
    ('sa_alias_store', 'data-lookup', False, {
        'main.py':
        'import string\n'
        'm = string\n'
        "m.digits = ''\n"
        'import random\n'
        'random.choice(m.digits)\n'
    }),
    ('sa_pkg_and_module_same_name', 'arithmetic', False, {
        'helper.py': 'def f():\n    return 1\n',
        'helper/__init__.py': 'def f():\n    return 1 / len(())\n',
        'main.py': 'import helper\nhelper.f()\n'
    }),
    ('sa_module_beats_namespace_dir', 'arithmetic', True, {
        'helper.py': 'def f():\n    return 1\n',
        'helper/other.py': 'def f():\n    return 1 / len(())\n',
        'main.py': 'import helper\nhelper.f()\n'
    }),
    ('sa_import_self_as_module', 'arithmetic', False, {
        'main.py': 'import main\nx = 1 / len(())\n'
    }),
    ('sa_dunder_main_import', 'arithmetic', False, {
        'main.py': 'import __main__\n'
    }),
    ('sa_cond_import_fallback', 'arithmetic', False, {
        'main.py':
        'try:\n'
        '    import helper\n'
        'except ImportError:\n'
        '    helper = None\n'
        'x = 1 / len(())\n'
    }),
    ('sa_enum_sunder_missing', 'data-lookup', False, {
        'main.py':
        'import enum\n'
        'class E(enum.Enum):\n'
        '    A = 1\n'
        '    @classmethod\n'
        '    def _missing_(cls, value):\n'
        '        return [][0]\n'
        'E(5)\n'
    }),
    ('sa_dataclass_post_init_subscript', 'data-lookup', False, {
        'main.py':
        'from dataclasses import dataclass\n'
        '@dataclass\n'
        'class P:\n'
        '    x: int\n'
        '    def __post_init__(self):\n'
        '        [][0]\n'
        'P(1)\n'
    }),
    ('sa_user_exception_base_lib', 'arithmetic', True, {
        'main.py': "class MyError(Exception):\n    pass\nx = MyError('a')\n"
    }),
    ('sa_type_compare_ctl', 'data-lookup', True, {
        'main.py': 'd = {}\nb = type(d) == type([])\nn = type(d).__name__\n'
    }),
    ('sa_type_assigned', 'data-lookup', False, {
        'main.py': 't = type\nx = t({})\n'
    }),
    ('sa_type_in_call_arg', 'data-lookup', False, {
        'main.py': 'd = {}\nx = list(type(d))\n'
    }),
    ('atk2_a1_alias_rebound_data', 'data-lookup', False, {
        'helper.py': 'def choice(x):\n    return 0\n',
        'main.py': 'import random\nimport mod1\n\nmod1.h = random\nmod1.go()\n',
        'mod1.py': 'import helper as h\n\ndef go():\n    return h.choice([])\n'
    }),
    ('atk2_a2_alias_rebound_arith', 'arithmetic', False, {
        'helper.py': 'def log(x, b):\n    return 0\n',
        'main.py': 'import math\nimport mod1\n\nmod1.h = math\nmod1.go()\n',
        'mod1.py': 'import helper as h\n\ndef go():\n    return h.log(10, 1)\n'
    }),
    ('atk2_a3_plain_import_rebound_data', 'data-lookup', False, {
        'helper.py': 'def choice(x):\n    return 0\n',
        'main.py': 'import random\nimport mod1\n\nmod1.helper = random\nmod1.go()\n',
        'mod1.py': 'import helper\n\ndef go():\n    return helper.choice([])\n'
    }),
    ('atk2_a4_from_import_submodule_rebound_data', 'data-lookup', False, {
        'main.py': 'import random\nimport mod1\n\nmod1.helper = random\nmod1.go()\n',
        'mod1.py': 'from pkg import helper\n\ndef go():\n    return helper.choice([])\n',
        'pkg/__init__.py': '',
        'pkg/helper.py': 'def choice(x):\n    return 0\n'
    }),
    ('atk2_a5_alias_rebound_heappop', 'data-lookup', False, {
        'helper.py': 'def heappop(x):\n    return 0\n',
        'main.py': 'import heapq\nimport mod1\n\nmod1.h = heapq\nmod1.go()\n',
        'mod1.py': 'import helper as h\n\ndef go():\n    return h.heappop([])\n'
    }),
    ('atk2_b1_pkg_init_binds_sub_data', 'data-lookup', False, {
        'main.py': 'import pkg\nimport other\n\npkg.sub.choice([])\n',
        'other.py': 'def never():\n    import pkg.sub\n',
        'pkg/__init__.py': 'import random as sub\n',
        'pkg/sub.py': 'def choice(x):\n    return 0\n'
    }),
    ('atk2_b2_pkg_init_binds_sub_arith', 'arithmetic', False, {
        'main.py': 'import pkg\nimport other\n\npkg.sub.log(10, 1)\n',
        'other.py': 'def never():\n    import pkg.sub\n',
        'pkg/__init__.py': 'import math as sub\n',
        'pkg/sub.py': 'def log(x, b):\n    return 0\n'
    }),
    ('atk2_c1_counter_delitem_hash', 'data-lookup', False, {
        'main.py':
        'from collections import Counter\n'
        '\n'
        'class K:\n'
        '    def __init__(self):\n'
        '        self.n = 0\n'
        '\n'
        '    def __hash__(self):\n'
        '        self.n += 1\n'
        '        return 0 if self.n < 3 else 1\n'
        '\n'
        'k = K()\n'
        'c = Counter({k: 0})\n'
        'c -= Counter()\n'
    }),
    ('atk2_c2_counter_delitem_hash_iand', 'data-lookup', False, {
        'main.py':
        'from collections import Counter\n'
        '\n'
        'class K:\n'
        '    def __init__(self):\n'
        '        self.n = 0\n'
        '\n'
        '    def __hash__(self):\n'
        '        self.n += 1\n'
        '        return 0 if self.n < 3 else 1\n'
        '\n'
        'c = Counter({K(): -1})\n'
        'c += Counter()\n'
    }),
    ('atk2_a6_star_import_rebinds_alias', 'data-lookup', False, {
        'helper.py': 'def choice(x):\n    return 0\n',
        'main.py': 'import mod1\n\nmod1.go()\n',
        'mod1.py':
        'import helper as h\nfrom other import *\n\ndef go():\n    return h.choice([])\n',
        'other.py': 'import random as h\n'
    }),
    ('atk2_a7_from_import_after_pkg_store', 'data-lookup', False, {
        'main.py':
        'import random\n'
        'import pkg\n'
        '\n'
        'pkg.sub = random\n'
        'from pkg import sub\n'
        '\n'
        'sub.choice([])\n',
        'pkg/__init__.py':
        '',
        'pkg/sub.py':
        'def choice(x):\n    return 0\n'
    }),
    ('atk2_b3_init_rebinds_loaded_sub', 'data-lookup', False, {
        'main.py': 'import pkg.sub\n\npkg.sub.choice([])\n',
        'pkg/__init__.py': 'from . import sub as _s\nimport random as sub\n',
        'pkg/sub.py': 'def choice(x):\n    return 0\n'
    }),
    ('atk2_b4_import_as_after_init_rebind', 'data-lookup', False, {
        'main.py': 'import pkg.sub as s\n\ns.choice([])\n',
        'pkg/__init__.py': 'from . import sub as _s\nimport random as sub\n',
        'pkg/sub.py': 'def choice(x):\n    return 0\n'
    }),
    ('atk2_b5_init_rebinds_loaded_sub_arith', 'arithmetic', False, {
        'main.py': 'import pkg.sub\n\npkg.sub.log(10, 1)\n',
        'pkg/__init__.py': 'from . import sub as _s\nimport math as sub\n',
        'pkg/sub.py': 'def log(x, b):\n    return 0\n'
    }),
    ('atk2_d1_self_import_list_store', 'data-lookup', False, {
        'main.py': 'import main\n\nmain.list = []\nprint(list[0])\n'
    }),
    ('atk2_d2_module_dict_store', 'data-lookup', False, {
        'main.py': 'import mod1\n\nmod1.dict = {}\nmod1.go()\n',
        'mod1.py': "def go():\n    return dict['k']\n"
    }),
    ('atk2_d3_module_tuple_store_aug', 'data-lookup', False, {
        'main.py': 'import mod1\n\nmod1.tuple = (1, 2)\nmod1.go()\n',
        'mod1.py': 'def go():\n    return tuple[3]\n'
    }),
    ('atk2_a8_self_import_alias_rebound', 'data-lookup', False, {
        'main.py':
        'import random\n'
        'import main as me\n'
        '\n'
        'def choice(x):\n'
        '    return 0\n'
        '\n'
        'me.me = random\n'
        'me.choice([])\n'
    }),
    ('atk2_h1_class_local_builtin_pow', 'arithmetic', False, {
        'main.py':
        'def f(pow=None):\n'
        '    class A:\n'
        '        x = pow(0, -1)\n'
        '        pow = None\n'
        '\n'
        'f()\n'
    }),
    ('atk2_h2_lambda_walrus_genexp', 'arithmetic', False, {
        'main.py':
        'def f():\n'
        '    list((lambda: (divmod := 1)) for _ in ())\n'
        '    return divmod(1, 0)\n'
        '\n'
        'f()\n'
    }),
    ('atk2_h3_typing_alias_store_universe', 'data-lookup', False, {
        'main.py': "import typing as t\n\nt.Dict = {}\nprint(t.Dict['k'])\n"
    }),
    ('atk2_h4_inherit_list', 'data-lookup', False, {
        'main.py': 'class L(list):\n    def keys(self):\n        return [0]\n\nprint(dict(L()))\n'
    }),
    ('atk2_h5_heappushpop_alias', 'data-lookup', False, {
        'main.py':
        'from heapq import heappushpop as h\n'
        '\n'
        'class X:\n'
        '    def __lt__(self, o):\n'
        '        hp.clear()\n'
        '        return True\n'
        '\n'
        'hp = [X()]\n'
        'h(hp, X())\n'
    }),
    ('atk2_h6_underscore_library_attr', 'data-lookup', False, {
        'main.py': 'import random\n\nrandom._inst.choice([])\n'
    }),
    ('atk2_h7_namespace_dir', 'data-lookup', False, {
        'main.py': 'import ns.mod\n',
        'ns/mod.py': 'import random\nrandom.choice([])\n'
    }),
    ('atk2_h8_three_arg_type_kw', 'data-lookup', False, {
        'main.py': "T = type('T', (list,), dict(keys=lambda self: [0]))\nprint(dict(T()))\n"
    }),
    ('atk2_n1_control_safe', 'data-lookup', True, {
        'main.py': 'x = [1, 2]\nx.append(3)\nprint(len(x))\n'
    }),
    ('atk2_n2_control_direct_choice', 'data-lookup', False, {
        'main.py': 'import random\n\nrandom.choice([])\n'
    }),
    ('atk3_a_wraps_random_normalvariate', 'arithmetic', False, {
        'main.py':
        '# family: arithmetic\n'
        'import functools\n'
        'import random\n'
        'class Src:\n'
        '    def random(self):\n'
        '        return 1.0\n'
        'r = random.Random(1)\n'
        'functools.wraps(Src(), assigned=("random",), updated=())(r)\n'
        'r.normalvariate()\n'
    }),
    ('atk3_d_base_class_attr_dict', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'class NS:\n'
        '    ABC = dict\n'
        'class D(NS.ABC):\n'
        '    def keys(self):\n'
        '        return ["x"]\n'
        '    def __iter__(self):\n'
        '        return iter(["x"])\n'
        'd = {}\n'
        'd.update(D())\n'
    }),
    ('atk3_d_base_class_attr_list', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'class NS:\n'
        '    Enum = list\n'
        'class L(NS.Enum):\n'
        '    def keys(self):\n'
        '        return [5]\n'
        'd = {}\n'
        'd.update(L())\n'
    }),
    ('atk3_d_base_collections_deque_alias', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'from collections import deque as Exception\n'
        'class Q(Exception):\n'
        '    def keys(self):\n'
        '        return [5]\n'
        'd = {}\n'
        'd.update(Q())\n'
    }),
    ('atk3_d_base_typing_alias_list', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'from typing import List as Enum\n'
        'class L(Enum):\n'
        '    def keys(self):\n'
        '        return [5]\n'
        'd = {}\n'
        'd.update(L())\n'
    }),
    ('atk3_d_base_typing_dict_alias_iter', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'from typing import Dict as ABC\n'
        'class D(ABC):\n'
        '    def keys(self):\n'
        '        return ["x"]\n'
        '    def __iter__(self):\n'
        '        return iter(["x"])\n'
        'd = dict(D())\n'
    }),
    ('atk3_d_enum_attr_del_member', 'data-lookup', False, {
        'main.py': '# family: data-lookup\nimport enum\nclass M(enum.Enum):\n    A = 1\n    del A\n'
    }),
    ('atk3_d_enum_attr_dict_ctor', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'import enum\n'
        'class M(enum.IntEnum):\n'
        '    A = 1\n'
        '    @classmethod\n'
        '    def keys(cls):\n'
        '        return ["B"]\n'
        'd = dict(M)\n'
    }),
    ('atk3_d_enum_attr_dict_update', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'import enum\n'
        'class M(enum.Enum):\n'
        '    A = 1\n'
        '    @classmethod\n'
        '    def keys(cls):\n'
        '        return ["nope"]\n'
        'd = {}\n'
        'd.update(M)\n'
    }),
    ('atk3_d_enum_functional_strenum', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'import enum\n'
        'class S(enum.StrEnum):\n'
        '    A = "x"\n'
        'N = enum.Enum("N", S)\n'
    }),
    ('atk3_d_wraps_getitem_dict', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'import functools\n'
        'class S:\n'
        '    def keys(self):\n'
        '        return ["a"]\n'
        'functools.wraps({}, assigned=("__getitem__",), updated=())(S)\n'
        'x = {}\n'
        'x.update(S())\n'
    }),
    ('atk3_d_wraps_getitem_list', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'import functools\n'
        'class S:\n'
        '    def keys(self):\n'
        '        return [0]\n'
        'functools.wraps([], assigned=("__getitem__",), updated=())(S)\n'
        'x = {}\n'
        'x.update(S())\n'
    }),
    ('atk3_a_mod_class_scope_import', 'arithmetic', False, {
        'helper.py':
        'D = 2\ndef div(x):\n    return x / D\n',
        'main.py':
        '# family: arithmetic\n'
        'import enum\n'
        'import functools\n'
        'import helper\n'
        'class C:\n'
        '    import helper\n'
        'functools.wraps(enum.IntEnum("X", [("D", 0)]), assigned=("D",), '
        'updated=())(C.helper)\n'
        'helper.div(1)\n'
    }),
    ('atk3_a_mod_default_dict_copy', 'arithmetic', False, {
        'helper.py':
        'D = 2\ndef div(x):\n    return x / D\n',
        'main.py':
        '# family: arithmetic\n'
        'import enum\n'
        'import functools\n'
        'import helper\n'
        'def g():\n'
        '    import helper\n'
        '    functools.wraps(enum.IntEnum("X", [("D", 0)]))(helper)\n'
        'g()\n'
        'helper.div(1)\n'
    }),
    ('atk3_a_mod_from_pkg_local', 'arithmetic', False, {
        'main.py':
        '# family: arithmetic\n'
        'import enum\n'
        'import functools\n'
        'from pkg import helper\n'
        'def g():\n'
        '    from pkg import helper\n'
        '    functools.wraps(enum.IntEnum("X", [("D", 0)]), assigned=("D",), '
        'updated=())(helper)\n'
        'g()\n'
        'helper.div(1)\n',
        'pkg/__init__.py':
        '',
        'pkg/helper.py':
        'D = 2\ndef div(x):\n    return x / D\n'
    }),
    ('atk3_a_mod_func_local_import', 'arithmetic', False, {
        'helper.py':
        'D = 2\ndef div(x):\n    return x / D\n',
        'main.py':
        '# family: arithmetic\n'
        'import enum\n'
        'import functools\n'
        'import helper\n'
        'def g():\n'
        '    import helper\n'
        '    functools.wraps(enum.IntEnum("X", [("D", 0)]), assigned=("D",), '
        'updated=())(helper)\n'
        'g()\n'
        'helper.div(1)\n'
    }),
    ('atk3_a_mod_func_local_return', 'arithmetic', False, {
        'helper.py':
        'D = 2\ndef div(x):\n    return x / D\n',
        'main.py':
        '# family: arithmetic\n'
        'import enum\n'
        'import functools\n'
        'import helper\n'
        'def g():\n'
        '    import helper\n'
        '    return helper\n'
        'functools.wraps(enum.IntEnum("X", [("D", 0)]), assigned=("D",), '
        'updated=())(g())\n'
        'helper.div(1)\n'
    }),
    ('atk3_a_mod_local_import_only', 'arithmetic', False, {
        'helper.py':
        'D = 2\ndef div(x):\n    return x / D\n',
        'main.py':
        '# family: arithmetic\n'
        'import enum\n'
        'import functools\n'
        'def g():\n'
        '    import helper\n'
        '    functools.wraps(enum.IntEnum("X", [("D", 0)]), assigned=("D",), '
        'updated=())(helper)\n'
        '    return helper.div(1)\n'
        'g()\n'
    }),
    ('atk3_a_wraps_decorator_random', 'arithmetic', False, {
        'main.py':
        '# family: arithmetic\n'
        'import functools\n'
        'import random\n'
        'class Src:\n'
        '    def random(self):\n'
        '        return 1.0\n'
        'r = random.Random(1)\n'
        'r2 = functools.wraps(Src(), assigned=("random",), updated=())(r)\n'
        'x = r2.normalvariate(0.0, 1.0)\n'
    }),
    ('atk3_d_base_func_return_attr', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'class Box:\n'
        '    ABC = list\n'
        'def get():\n'
        '    return Box\n'
        'class L(get().ABC):\n'
        '    def keys(self):\n'
        '        return [3]\n'
        'd = {}\n'
        'd.update(L())\n'
    }),
    ('atk3_d_base_list_sort_cmp', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'class NS:\n'
        '    ABC = list\n'
        'class L(NS.ABC):\n'
        '    def keys(self):\n'
        '        return [0]\n'
        'd = dict(L())\n'
    }),
    ('atk3_d_base_typing_ordereddict_alias', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'from typing import DefaultDict as Protocol\n'
        'class D(Protocol):\n'
        '    def keys(self):\n'
        '        return ["x"]\n'
        '    def __iter__(self):\n'
        '        return iter(["x"])\n'
        'd = {}\n'
        'd.update(D())\n'
    }),
    ('atk3_d_enum_attr_dict_ctor_plain', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'import enum\n'
        'class M(enum.Enum):\n'
        '    A = 1\n'
        '    @classmethod\n'
        '    def keys(cls):\n'
        '        return ["A", "B"]\n'
        'd = dict(M)\n'
    }),
    ('atk3_d_enum_attr_dict_update_kw', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'import enum\n'
        'class M(enum.Enum):\n'
        '    A = 1\n'
        '    @classmethod\n'
        '    def keys(cls):\n'
        '        return ["nope"]\n'
        'd = {}\n'
        'd.update(M, x=1)\n'
    }),
    ('atk3_d_enum_del_member_flag', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'import enum\n'
        'class M(enum.Flag):\n'
        '    A = 1\n'
        '    B = 2\n'
        '    del B\n'
    }),
    ('atk3_d_enum_functional_dict_keys_strenum', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'import enum\n'
        'class S(enum.StrEnum):\n'
        '    A = "B"\n'
        '    B = "zz"\n'
        'N = enum.Enum("N", S)\n'
    }),
    ('atk3_d_wraps_class_getitem_enum', 'data-lookup', False, {
        'main.py':
        '# family: data-lookup\n'
        'import enum\n'
        'import functools\n'
        'class E(enum.Enum):\n'
        '    A = 1\n'
        'class S:\n'
        '    def keys(self):\n'
        '        return ["zz"]\n'
        'functools.wraps(E, assigned=("__getitem__",), updated=())(S)\n'
        'd = dict(S())\n'
    }),
]
