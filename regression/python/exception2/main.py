x:int = 0
try:
    # some code
    x = x + 1
except ValueError:      # name is null - no variable binding
    pass
    
except TypeError as e:  # name is "e" - has variable binding  
    pass
