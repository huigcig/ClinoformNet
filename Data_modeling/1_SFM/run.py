import numpy as np
from badlands.model import Model as badlandsModel

for ip in range(1000):
    dir =  './Models/xmldirs/basin_demo'+str(num)+'.xml'
    print(dir)
    print("\n")
    model = badlandsModel()
    model.load_xml(dir)
    model.run_to_time(20000000)
    print("********" + dir +' is done ********')
    print("\n")

    

