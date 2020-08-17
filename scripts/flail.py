#!/usr/bin/env python3

import platform
from pathlib import Path
import subprocess
import sys
import re


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

    def __init__(self, filepath: str):
        WINDOWS = platform.system() == 'Windows'
        self._cat = 'cat.exe' if WINDOWS else 'cat'
        self._od = 'od.exe' if WINDOWS else 'od'
        self.filepath = filepath

    def od_cli_command(self):
        return f'{self._od} -v -t u1 {self.filepath}'

    def cat_cli_command(self):
        return f'{self._cat} {self.filepath}'

    def obtain_var_name(self):
        obj = Path(self.filepath)
        return obj.name.replace('.hs', '_buf')

    def _step_2(self, content: str):
        return Flail.REGEX_REMOVE_ADDR.sub('', content)

    def _step_3_4(self, content: str):
        return Flail.REGEX_MULTI_SPACE.sub(',', content)

    def _step_5(self, content: str):
        if Flail.REGEX_ADD_COMMA_END.match(content):
            return content + ','
        return '0\n'

    def _step_6(self, content, output):
        # its hard to use '{' '}' in F-Strings
        left_curly = '{'
        right_curly = '}'
        with open(output, 'w') as f:
            f.write(f'char {self.obtain_var_name()} [] = {left_curly}')
            f.writelines(content)
            f.write(f'{right_curly};')
            f.write(
                f'unsigned int {self.obtain_var_name()}_size = sizeof({self.obtain_var_name()});')

    def run(self, output_file: str):
        ps = subprocess.Popen(self.cat_cli_command().split(
        ), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        output = subprocess.check_output(
            self.od_cli_command().split(), stdin=ps.stdout).decode().splitlines()

        step_2 = [self._step_2(x) for x in output]
        step_3_4 = [self._step_3_4(x) for x in step_2]
        step_5 = [self._step_5(x) for x in step_3_4]

        self._step_6(step_5, output_file)


def main():
    if len(sys.argv) != 3:
        raise ValueError("Program expects <input> <output> arguments")

    filepath = sys.argv[1]
    output = sys.argv[2]
    obj = Flail(filepath)
    obj.run(output)


if __name__ == "__main__":
    main()
