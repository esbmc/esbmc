#!/usr/bin/env python3

import os.path
import sys
import subprocess
import tempfile

from multiprocessing import Process
import time

import shutil

class Csmith:
    """Wrapper for Csmith"""
    def __init__(self, csmith: str, csmith_inc: str, csmith_args=""):
        if not os.path.exists(csmith):
            raise ValueError(f'{csmith} does not exist!')
        if not os.path.exists(os.path.join(csmith_inc, "csmith.h")):
            raise ValueError(f'{csmith_inc} does not contain csmith.h!')
        self.csmith = csmith
        self.csmith_inc = csmith_inc
        self.csmith_args = csmith_args

    def generate_c_file(self, output):
        """Run csmith using `args` and saving in `output`"""
        cmd = f'{self.csmith} {self.csmith_args}'
        cmd_out = subprocess.check_output(cmd.split()).decode()

        with open(output, 'w') as f:
            f.write(cmd_out)

class ESBMC:
    """Wrapper for ESBMC"""

    def __init__(self, esbmc: str, esbmc_args=""):
        if not os.path.exists(esbmc):
            raise ValueError(f'{esbmc} does not exist!')
        self.esbmc = esbmc
        self.esbmc_args = esbmc_args

    def run(self, csmith_inc, csmith_file="", timeout=10) -> int:
        """Run esbmc with `args` and return exit code"""
        cmd = f'{self.esbmc} -I{csmith_inc} {csmith_file} {self.esbmc_args}'
        try:
            print("Running " + cmd)
            ps = subprocess.run(cmd.split(), timeout=int(timeout), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(ps)
            return ps.returncode
        except Exception as exception:
            print(exception.__class__.__name__) # Expect to see Timedout
            return 0

class Driver:
    """Main driver"""

    def __init__(self, csmith: Csmith, esbmc: ESBMC, esbmc_timeout=10):
        self.csmith = csmith
        self.esbmc = esbmc
        self.esbmc_timeout = esbmc_timeout

    def _main(self):
        if not os.path.exists('csmith-tests/'):
            os.makedirs('csmith-tests')
        counter = 0
        while True:
            counter = counter + 1
            c_file = "csmith-tests/" + str(counter) + ".c"
            # 1. Generate C file
            self.csmith.generate_c_file(c_file)

            # 2. Run ESBMC
            res = self.esbmc.run(self.csmith.csmith_inc, c_file, self.esbmc_timeout)

            # Check if an error was found
            # FUTURE: For max-coverage we probably just have to remove this 'if'
            if res != 0:                
                print("Found Error!")
                if not os.path.exists('csmith-error/'):
                    os.makedirs('csmith-error')
                shutil.copyfile(c_file, "csmith-error/error.c")
                shutil.copyfile(os.path.join(self.csmith.csmith_inc, "csmith.h"), "csmith-error/csmith.h")
                with open("csmith-error/desc") as f:
                    f.write(self.esbmc.esbmc_args)
                    f.write(res)
                return


    def run(self, timeout=100):
        """Start driver with defined timeout"""
        ps = Process(target=self._main)
        ps.start()
        ps.join(timeout=int(timeout))
        ps.terminate()

def main():
    print("Running csmith over esbmc...")
    csmith = sys.argv[1]
    csmith_inc = sys.argv[2]
    csmith_args = sys.argv[3]

    esbmc = sys.argv[4]
    esbmc_args = sys.argv[5]
    esbmc_timeout = sys.argv[6]

    driver_timeout = sys.argv[7]

    csmith_obj = Csmith(csmith, csmith_inc, csmith_args)
    esbmc_obj = ESBMC(esbmc, esbmc_args)

    driver = Driver(csmith_obj, esbmc_obj, esbmc_timeout)
    driver.run(driver_timeout)

    print("Done")

if __name__ == "__main__":
    main()
