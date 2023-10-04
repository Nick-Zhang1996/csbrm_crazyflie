import pickle
import numpy as np
import scipy.io

for i in range(1,21):
    text = str(i)
    output = open('./logs/final' + text + '.p','rb')
    data = pickle.load(output)
    output.close()

    data = np.array(data)
    print("log length %.1f seconds"%(data.shape[0]/119.88))

    skip = 0
    t = data[skip:,0] - data[0,0]
    x = data[skip:,1]
    y = data[skip:,2]
    z = data[skip:,3]
    rx = data[skip:,4]
    ry = data[skip:,5]
    rz = data[skip:,6]
    # convert vicon frame to planner frame
    # x_p = x*0.98 + 0.02
    x_p = x
    y_p = -y
    z_p = -z+1.2
    x = x_p
    y = y_p
    z = z_p

    scipy.io.savemat('./Matlabdata/final' + text + '.mat', mdict={'t': t, 'x_p': x_p, 'y_p': y_p, 'z_p': z_p})