#!/usr/bin/env python3

import platform
import unittest
from pathlib import Path
import subprocess
import sys
import re
import argparse
import os
import textwrap


class Flail:
    """
        The flailng generates the byte representation of the input file
        by converting its content into a char array. The resulting array will be written in a *.c file.
        The input file can be of any file type. For an input file of binary content, hexdump it
        and you'll see the matching between the output and each element in the result array.
        For input files that contain any text content, see example below:

        Example:

        Step 1: Object Dump
            #ifndef __ESBMC_HEADERS_STDARG_H_
            #define __ESBMC_HEADERS_STDARG_H_
            ...
            the byte representation of the above directives is:
            (matching the ASCII code of each character)
            ...
            0000000  35 105 102 110 100 101 102  32  95  95  69  83  66  77  67  95
            0000020  72  69  65  68  69  82  83  95  83  84  68  65  82  71  95  72
            0000040  95  13  10  35 100 101 102 105 110 101  32  95  95  69  83  66

        Step 2: Remove Address header
            ...
            input:
            ...
            0000000  35 105 102 110 100 101 102  32  95  95  69  83  66  77  67  95
            0000020  72  69  65  68  69  82  83  95  83  84  68  65  82  71  95  72
            0000040  95  13  10  35 100 101 102 105 110 101  32  95  95  69  83  66
            ...
            output:
            ...
            35 105 102 110 100 101 102  32  95  95  69  83  66  77  67  95
            72  69  65  68  69  82  83  95  83  84  68  65  82  71  95  72
            95  13  10  35 100 101 102 105 110 101  32  95  95  69  83  66

        Step 3: Replace multiple spaces with a single-one
            ...
            input:
            ...
            35 105 102 110 100 101 102  32  95  95  69  83  66  77  67  95
            72  69  65  68  69  82  83  95  83  84  68  65  82  71  95  72
            95  13  10  35 100 101 102 105 110 101  32  95  95  69  83  66
            ...
            output:
            ...
            35 105 102 110 100 101 102 32 95 95 69 83 66 77 67 95
            72 69 65 68 69 82 83 95 83 84 68 65 82 71 95 72
            95 13 10 35 100 101 102 105 110 101 32 95 95 69 83 66

        Step 4: Replace spaces with a comma
            ...
            input:
            ...
            35 105 102 110 100 101 102 32 95 95 69 83 66 77 67 95
            72 69 65 68 69 82 83 95 83 84 68 65 82 71 95 72
            95 13 10 35 100 101 102 105 110 101 32 95 95 69 83 66
            ...
            output:
            ...
            35,105,102,110,100,101,102,32,95,95,69,83,66,77,67,95
            72,69,65,68,69,82,83,95,83,84,68,65,82,71,95,72
            95,13,10,35,100,101,102,105,110,101,32,95,95,69,83,66

        Step 5: Adds a comma at the end
            ...
            input:
            ...
            35,105,102,110,100,101,102,32,95,95,69,83,66,77,67,95
            72,69,65,68,69,82,83,95,83,84,68,65,82,71,95,72
            95,13,10,35,100,101,102,105,110,101,32,95,95,69,83,66
            ...
            output:
            ...
            35,105,102,110,100,101,102,32,95,95,69,83,66,77,67,95,
            72,69,65,68,69,82,83,95,83,84,68,65,82,71,95,72,
            95,13,10,35,100,101,102,105,110,101,32,95,95,69,83,66,

        Step 6: Create a C header containing the output of Step 5 as a char array
        """

    REGEX_REMOVE_ADDR = re.compile(r'^[0-9]+( +)?')
    REGEX_MULTI_SPACE = re.compile(r' +')
    REGEX_ADD_COMMA_END = re.compile(r'.*[0-9]$')

    def __init__(self, filepath: str, prefix : str = ''):
        self.filepath = filepath
        self.prefix = prefix

    def custom_od(self):
        '''
        Generates octal representation of a file
        '''
        chars_per_line = 16
        lines = []
        with open(self.filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(chars_per_line), b''):
                line = [ int(x,16) for x in chunk.hex(' ').split(' ')]
                lines.append(' '.join(map(str, line)))
        return lines


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
        step_2 = self.custom_od()
        step_3_4 = [self._step_3_4(x) for x in step_2]
        step_5 = [self._step_5(x) for x in step_3_4]

        self._step_6(step_5, output_file, header, macro)


def parse_args(argv):

    p = argparse.ArgumentParser(prog=argv[0],
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            Try the following examples:

            Ex.1 Convert a list of C operational models (see last line of the CMD below) into char arrays.
                 Write the resulting array into libc.c and libc.h
                 CMD:
                    python3 flail.py \\
                        --macro ESBMC_FLAIL \\
                        --prefix esbmc_libc_ \\
                        -o ./libc.c \\
                        --header ./libc.h \\
                        src/c2goto/library/ctype.c

            Ex.2 Convert a goto binary file (see last line of the CMD below) into a char array
                 After building ESBMC, find clib32.goto from the build tree
                 CMD:
                    python3 flail.py -o ./clib32.c \\
                            esbmc/build/src/c2goto/clibd32.goto
            NB:
            --macro MACRO is only meaningful in combination with --header;
            if specified, the header file will contain invocations of MACRO(body, size, fpath) for each input file where body is the name of
            the extern char[] symbol defining the binary content of the input file, size is its size and fpath is the path to the input file as passed to this script.
            If MACRO is not defined, the header will define it before the invocations to declare the body and size symbols as extern to facilitate usage
            of the header in a C file
            '''))
    p.add_argument('-p', '--prefix', type=str, default='',
                   help='prefix for C symbols (default: empty)')
    p.add_argument('--macro', type=str, default=None,
                   help='define MACRO in the header file (only meaningful in combination with --header)')
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
