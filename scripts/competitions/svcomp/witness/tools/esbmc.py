# This file is part of BenchExec, a framework for reliable benchmarking:
# https://github.com/sosy-lab/benchexec
#
# SPDX-FileCopyrightText: 2007-2020 Dirk Beyer <https://www.sosy-lab.org>
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from benchexec.tools.sv_benchmarks_util import (
    get_data_model_from_task,
    handle_witness_of_task,
    TaskFilesConsidered,
    ILP32,
    LP64,
)
import benchexec.tools.template
import benchexec.result as result
import decimal


class Tool(benchexec.tools.template.BaseTool2):
    """
    This class serves as tool adaptor for ESBMC
    """

    def executable(self, tool_locator):
        return tool_locator.find_executable("esbmc-wrapper.py")

    def working_directory(self, executable):
        executableDir = os.path.dirname(executable)
        return executableDir

    def version(self, executable):
        return self._version_from_tool(executable, "-v")

    def name(self):
        return "ESBMC"

    def project_url(self):
        return "http://www.esbmc.org/"

    def cmdline(self, executable, options, task, rlimits):
        data_model_param = get_data_model_from_task(task, {ILP32: "32", LP64: "64"})
        if data_model_param and "--arch" not in options:
            options += ["--arch", data_model_param]

        input_files, witness_options = handle_witness_of_task(
            task,
            options,
            "--witness",
            TaskFilesConsidered.SINGLE_INPUT_FILE,
        )

        return (
            [executable]
            + ["-p", task.property_file]
            + options
            + input_files
            + witness_options
        )

    def determine_result(self, run):
        status = result.RESULT_UNKNOWN

        if run.output.any_line_contains("FALSE_DEREF"):
            status = result.RESULT_FALSE_DEREF
        elif run.output.any_line_contains("FALSE_FREE"):
            status = result.RESULT_FALSE_FREE
        elif run.output.any_line_contains("FALSE_MEMTRACK"):
            status = result.RESULT_FALSE_MEMTRACK
        elif run.output.any_line_contains("FALSE_MEMCLEANUP"):
            status = result.RESULT_FALSE_MEMCLEANUP
        elif run.output.any_line_contains("FALSE_OVERFLOW"):
            status = result.RESULT_FALSE_OVERFLOW
        elif run.output.any_line_contains("FALSE_TERMINATION"):
            status = result.RESULT_FALSE_TERMINATION
        elif run.output.any_line_contains("FALSE_DATARACE"):
            status = result.RESULT_FALSE_DATARACE
        elif run.output.any_line_contains("FALSE"):
            status = result.RESULT_FALSE_REACH
        elif run.output.any_line_contains("TRUE"):
            status = result.RESULT_TRUE_PROP
        elif run.output.any_line_contains("DONE"):
            status = result.RESULT_DONE

        if status == result.RESULT_UNKNOWN:
            if run.was_timeout:
                status = result.RESULT_TIMEOUT
            elif not run.output.any_line_contains("Unknown"):
                status = "ERROR"

        return status

    def get_value_from_output(self, output, identifier):
        regex = re.compile(identifier)
        matches = []

        # Match first element of each line
        for line in output:
            match = regex.search(line.strip())
            if match and len(match.groups()) >= 1:
                matches.append(match.group(1))

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return sum(decimal.Decimal(value) for value in matches)
        return None
