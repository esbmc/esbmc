#!/usr/bin/env python3

import platform
import unittest
from pathlib import Path
import subprocess
import sys
import re
import argparse
import os


class Flail:
    """
        The flaing consists in converting a file to a C header containing all the contents
        as an uint array

        Example:

        Step 1: Object Dump

            #ifndef __ESBMC_HEADERS_STDARG_H_
            #define __ESBMC_HEADERS_STDARG_H_
            ...
                    vvv
            0000000  35 105 102 110 100 101 102  32  95  95  69  83  66  77  67  95
            0000020  72  69  65  68  69  82  83  95  83  84  68  65  82  71  95  72
            0000040  95  13  10  35 100 101 102 105 110 101  32  95  95  69  83  66
            ...

        Step 2: Remove Address header
            0000000  35 105 102 110 100 101 102  32  95  95  69  83  66  77  67  95
            0000020  72  69  65  68  69  82  83  95  83  84  68  65  82  71  95  72
            0000040  95  13  10  35 100 101 102 105 110 101  32  95  95  69  83  66
            ...
                    vvv
            35 105 102 110 100 101 102  32  95  95  69  83  66  77  67  95
            72  69  65  68  69  82  83  95  83  84  68  65  82  71  95  72
            95  13  10  35 100 101 102 105 110 101  32  95  95  69  83  66

        Step 3: Replace multiple spaces with a single-one
            35 105 102 110 100 101 102  32  95  95  69  83  66  77  67  95
            72  69  65  68  69  82  83  95  83  84  68  65  82  71  95  72
            95  13  10  35 100 101 102 105 110 101  32  95  95  69  83  66
                    vvv
            35 105 102 110 100 101 102 32 95 95 69 83 66 77 67 95
            72 69 65 68 69 82 83 95 83 84 68 65 82 71 95 72
            95 13 10 35 100 101 102 105 110 101 32 95 95 69 83 66

        Step 4: Replace spaces with a comma
            35 105 102 110 100 101 102 32 95 95 69 83 66 77 67 95
            72 69 65 68 69 82 83 95 83 84 68 65 82 71 95 72
            95 13 10 35 100 101 102 105 110 101 32 95 95 69 83 66
                    vvv
            35,105,102,110,100,101,102,32,95,95,69,83,66,77,67,95
            72,69,65,68,69,82,83,95,83,84,68,65,82,71,95,72
            95,13,10,35,100,101,102,105,110,101,32,95,95,69,83,66

        Step 5: Adds a comma at the end
            35,105,102,110,100,101,102,32,95,95,69,83,66,77,67,95
            72,69,65,68,69,82,83,95,83,84,68,65,82,71,95,72
            95,13,10,35,100,101,102,105,110,101,32,95,95,69,83,66
                    vvv
            35,105,102,110,100,101,102,32,95,95,69,83,66,77,67,95,
            72,69,65,68,69,82,83,95,83,84,68,65,82,71,95,72,
            95,13,10,35,100,101,102,105,110,101,32,95,95,69,83,66,

        Step 6: Create a C header
        """

    REGEX_REMOVE_ADDR = re.compile(r'^[0-9]+( +)?')
    REGEX_MULTI_SPACE = re.compile(r' +')
    REGEX_ADD_COMMA_END = re.compile(r'.*[0-9]$')

    def __init__(self, filepath: str, prefix : str = ''):
        WINDOWS = platform.system() == 'Windows'
        self._cat = 'cat.exe' if WINDOWS else 'cat'
        self._od = 'od.exe' if WINDOWS else 'od'
        self.filepath = filepath
        self.prefix = prefix

    def od_cli_command(self):
        return f'{self._od} -v -t u1 {self.filepath}'

    def cat_cli_command(self):
        return f'{self._cat} {self.filepath}'

    def obtain_var_name(self):
        obj = Path(self.filepath)
        return self.prefix + (obj.name.replace('.hs', '_buf')
                                      .replace('.h', '_buf')
                                      .replace('.c', '_buf')
                                      .replace('.goto', '_buf')
                                      .replace('.txt', '_buf')
                                      .replace('buildidobj', 'buildidstring')
                                      .replace('-', '_'))

    def _step_2(self, content: str):
        return Flail.REGEX_REMOVE_ADDR.sub('', content)

    def _step_3_4(self, content: str):
        return Flail.REGEX_MULTI_SPACE.sub(',', content)

    def _step_5(self, content: str):
        if len(content) > 0 and content[-1] != ",":
            return content + ',\n'
        return content + '\n'

    def _step_6(self, content, output, header, macro : str):
        name = self.obtain_var_name()
        output.write('const char %s[] = {\n' % name)
        output.writelines(content)
        output.write('};\n')
        output.write('const unsigned int %s_size = sizeof(%s);\n' % (name, name))
        if header is not None:
            if macro is None:
                header.write('extern const char %s[];\n' % name)
                header.write('extern const unsigned int %s_size;\n' % name)
            else:
                header.write('%s(%s, %s_size, %s)\n' % (macro, name, name,
                                                        self.filepath))

    def run(self, output_file, header = None, macro : str = None):
        ps = subprocess.Popen(self.cat_cli_command().split(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        output = subprocess.check_output(self.od_cli_command().split(),
                                         stdin=ps.stdout
                                        ).decode().splitlines()

        step_2 = [self._step_2(x) for x in output]
        step_3_4 = [self._step_3_4(x) for x in step_2]
        step_5 = [self._step_5(x) for x in step_3_4]

        self._step_6(step_5, output_file, header, macro)


def parse_args(argv):
    p = argparse.ArgumentParser(prog=argv[0])
    p.add_argument('-p', '--prefix', type=str, default='',
                   help='prefix for C symbols (default: empty)')
    p.add_argument('--macro', type=str, default=None,
                   help='print MACRO invocation to stdout')
    p.add_argument('--header', type=str, default=None,
                   help='write header file containing "extern" declarations of '
                        'generated symbols')
    p.add_argument('-o', '--output', type=str, required=True)
    p.add_argument('input', type=str, nargs='+')
    return p.parse_args(argv[1:])


def main():
    args = parse_args(sys.argv)
    with open(args.output, 'w') as output:
        header = None
        if args.header:
            header = open(args.header, 'w')
            if args.macro:
                header.write('#ifndef %s\n' % args.macro)
                header.write('# define %s(body, size, ...)'
                             ' extern const char body[];'
                             ' extern const unsigned int size;\n' % args.macro)
                header.write('#endif\n')
        for infile in args.input:
            obj = Flail(infile, args.prefix)
            obj.run(output, header, args.macro)
        if header is not None:
            header.close()


if __name__ == "__main__":
    main()


# TESTS

# `python3 -m unittest flail`

class TestFlail(unittest.TestCase):
    # Since the objdump result may be system dependent, this will test the transformations

    OBJDUMP = ["0000000  35 105 102 110 100 101 102  32  95  95  69  83  66  77  67  95",
               "0000020  72  69  65  68  69  82  83  95  83  84  68  65  82  71  95  72",
               "0000040  95  13  10  35 100 101 102 105 110 101  32  95  95  69  83  66"]

    STEP_2_EXPECTED = ["35 105 102 110 100 101 102  32  95  95  69  83  66  77  67  95",
                       "72  69  65  68  69  82  83  95  83  84  68  65  82  71  95  72",
                       "95  13  10  35 100 101 102 105 110 101  32  95  95  69  83  66"]

    STEP_3_4_EXPECTED = ["35,105,102,110,100,101,102,32,95,95,69,83,66,77,67,95",
                         "72,69,65,68,69,82,83,95,83,84,68,65,82,71,95,72",
                         "95,13,10,35,100,101,102,105,110,101,32,95,95,69,83,66"]

    STEP_5_EXPECTED = ["35,105,102,110,100,101,102,32,95,95,69,83,66,77,67,95,\n",
                       "72,69,65,68,69,82,83,95,83,84,68,65,82,71,95,72,\n",
                       "95,13,10,35,100,101,102,105,110,101,32,95,95,69,83,66,\n"]

    def setUp(self):
        obj = Flail("")
        self.step_2 = [obj._step_2(x) for x in self.__class__.OBJDUMP]
        self.step_3_4 = [obj._step_3_4(x) for x in self.step_2]
        self.step_5 = [obj._step_5(x) for x in self.step_3_4]

    def test_step_2(self):
        self.assertEqual(self.step_2, self.__class__.STEP_2_EXPECTED)

    def test_step_3(self):
        self.assertEqual(self.step_3_4, self.__class__.STEP_3_4_EXPECTED)

    def test_step_5(self):
        self.assertEqual(self.step_5, self.__class__.STEP_5_EXPECTED)

    def test_variable_name_1(self):
        obj = Flail("a.h")
        expected = "a_buf"
        self.assertEqual(obj.obtain_var_name(), expected)

    def test_variable_name_2(self):
        obj = Flail("b.goto")
        expected = "b_buf"
        self.assertEqual(obj.obtain_var_name(), expected)

    def test_variable_name_3(self):
        obj = Flail("c.txt")
        expected = "c_buf"
        self.assertEqual(obj.obtain_var_name(), expected)

    def test_variable_name_4(self):
        obj = Flail("buildidobj")
        expected = "buildidstring"
        self.assertEqual(obj.obtain_var_name(), expected)

    def test_variable_name_1_prefix(self):
        obj = Flail("a.h", 'prefix_')
        expected = "prefix_a_buf"
        self.assertEqual(obj.obtain_var_name(), expected)

    def test_variable_name_2_prefix(self):
        obj = Flail("b.goto", 'prefix_')
        expected = "prefix_b_buf"
        self.assertEqual(obj.obtain_var_name(), expected)

    def test_variable_name_3_prefix(self):
        obj = Flail("c.txt", 'prefix_')
        expected = "prefix_c_buf"
        self.assertEqual(obj.obtain_var_name(), expected)

    def test_variable_name_4_prefix(self):
        obj = Flail("buildidobj", 'prefix_')
        expected = "prefix_buildidstring"
        self.assertEqual(obj.obtain_var_name(), expected)
