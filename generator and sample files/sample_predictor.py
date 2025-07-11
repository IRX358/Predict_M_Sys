# I made a sample predict program to carry on with my work so tht i dont have to wait till farzan's model completion.
#this gives some mock results in the name of predictions 

import sys

def sample_predict(file_path):
    #returning some mock value without actual evaluation of logic with CNN model
    #the CNN model logical code will be coming here
    # return f"Predicted result for {file_path} : Healthy"
    return f"Healthy"

if __name__=="__main__":
    file_path=sys.argv[1]
    print(sample_predict(file_path))