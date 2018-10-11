import numpy as np
import matplotlib.pyplot as plt

# training_set = np.load('s_allPoints.npy')
# training_set = np.asarray(training_set)
# test_val = np.ndarray.tolist(training_set[99])
# training_set = training_set.tolist()
# temp_val = np.array(1)
# for x in range(100):
#     new_vector = np.append(1, training_set[x])
#     "print new_vector"
#
# supervised_s_1 = np.load('s_1_points.npy')
# size_val = supervised_s_1.shape[0]
# "print (any(training_set[:] == [0.3405933, 0.65402327]).all(1))"
# bool_val = test_val in training_set
# print (test_val in training_set)
# "print (any(np.equal(training_set, [0.3405933, 0.65402327]).all(1)))"
# print training_set

#training_set = np.load('s_1_points.npy')
#plt.scatter(training_set[:, 0], training_set[:, 1], color='blue')
#plt.show()

training_set = np.load('initialWeights.npy')
training_set = np.asarray(training_set)
print training_set

"""
load_weights = np.load('ptaWeights.npy')
temp_vector = np.array([[1], [training_set[99][0]], [training_set[99][1]]])
print load_weights[0]
print "fuck"
print load_weights[1]
print "fuck"
print load_weights[2]
print load_weights.shape

print np.transpose(temp_vector)
print np.transpose(load_weights)

cal_out = np.dot(np.transpose(temp_vector)[0], np.transpose(load_weights)[0])
print ("outout val is/////////", cal_out)

"""
"""
training_set = np.load('s_allPoints.npy')
training_set = np.asarray(training_set)
print training_set[99]
print ("try to get", training_set[99][0])
print ("try to get", training_set[99][1])
temp_vector = np.array([[1], [training_set[99][0]], [training_set[99][1]]])
new_temp_vector = np.transpose(temp_vector)
print ("shape of the vector created is", temp_vector.shape)
print temp_vector
print ("*************************************************")
print ("*************************************************")
print ("shape of the transpose vector created is", new_temp_vector.shape)
print new_temp_vector"""

