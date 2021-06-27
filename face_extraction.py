from os import listdir
from numpy import asarray
from PIL import Image
from matplotlib import pyplot
from pylab import *
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

def load_faces(directory, n):
    model = MTCNN()
    faces = list()
    for filename in listdir(directory):
        if(filename==".DS_Store"):
            continue
        print("filename is ")
        print(filename)
        pixels = load_image("i2.jpeg")
        face = extract_face(model, pixels)
        if(face is None):
            continue

        faces.append(face)
        print(len(faces), face.shape)
        if(len(faces)>=n):
            break
    return asarray(faces)

def plot_faces(faces, n):
    for i in range(n*n):
        pyplot.subplot(n,n,1+i)
        pyplot.axis('off')
        pyplot.imshow(faces[i])
    pyplot.show()


directory = "musk/"
faces = load_faces(directory,1)
print("Loaded: ", faces.shape)
plot_faces(faces,1)
savez_compressed("musk.npz", faces)
