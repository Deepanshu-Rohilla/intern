from os import listdir
from numpy import asarray
from PIL import Image
from matplotlib import pyplot
from pylab import *
import matplotlib.pyplot as plt
from sklearn import decomposition
from mtcnn.mtcnn import MTCNN

def load_image(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    return pixels

def print1():
    print("LOL")

def extract_face(model, pixels, required_size=(80,80)):
    faces = model.detect_faces(pixels)
    if(len(faces)==0):
        return None
    x1,y1,width,height = faces[0]['box']
    x1,y1 = abs(x1), abs(y1)
    x2,y2 = x1 + width, y1+height
    face_pixels = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face_pixels)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def load_face(filename):
    model = MTCNN()
    faces = list()
    pixels = load_image(filename)
    face = extract_face(model, pixels)
    print(face.shape)
    return asarray(faces)

def plot_face(face):
    for i in range(1):
        pyplot.subplot(1,1,1+i)
        pyplot.axis('off')
        pyplot.imshow(face)
    pyplot.show()


filename = "i1.jpeg"
face = load_face(filename)
print(type(face))
print("Loaded: ")
# plt.plot(face)
# plt.show()
plot_face(face)
savez_compressed("single_musk.npz", face)