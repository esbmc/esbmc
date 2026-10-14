#!/bin/sh
# A local model solver that accepts the formula but answers with an
# unparseable line, standing in for a solver that crashes mid-reply.
printf 'this is not an s-expression )\n'
exec cat >/dev/null
