import caffe
import cv2

model = '/home/wogong/Models/caffe/bvlc_reference_caffenet/deploy.prototxt'
weights = '/home/wogong/Models/caffe/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(model, weights, 'test')

image = cv2.imread('../test.jpg')
res = net.forward({image})
print(res)
