import tinygrad
import inspect
   
   # Print all module contents
print(dir(tinygrad))
   
   # Search for LSTM in submodules
for name, obj in inspect.getmembers(tinygrad):
    if inspect.ismodule(obj):
        if 'LSTM' in dir(obj):
            print(f"Found LSTM in {name}")