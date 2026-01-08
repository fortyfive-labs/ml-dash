# from ml_dash import Experiment                                                                     
# with Experiment(prefix="scratch/test-run").run as exp:                                             
#     for i in range(10):                                                                            
# 	exp.metrics("train").log(step=i, loss=1.0/(i+1))                                           
#                                                                                                    
# Or using the auto-start singleton:                                                                 
                                                                                                   
from ml_dash.auto_start import dxp                                                                 
                                                                                                   
with dxp.run:                                                                                      
    for i in range(10):                                                                            
        dxp.metrics("train").log(step=i, loss=1.0/(i+1))    
