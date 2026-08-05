# A module never imports itself, so a variable may carry the module's own name.
# is_imported_module fell back to "does a module of this name exist on disk",
# and the file being converted always does, so the receiver resolved to the
# module and no method call on it dispatched (#6639).
import queue

main: queue.Queue = queue.Queue()
main.put(7)
x = main.get()
assert x == 7
