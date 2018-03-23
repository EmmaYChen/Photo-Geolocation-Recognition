# quiver engine: https://keplr-io.github.io/quiver/

from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')

from quiver_engine.server import launch
launch(model, input_folder='./img', port = 7000)

type(model)