from os import listdir
from numpy import asarray
from PIL import Image
from matplotlib import pyplot

def load_image(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    return pixels

def print1():
    print("LOL")

