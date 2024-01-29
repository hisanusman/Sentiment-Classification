import pickle
import numpy as np
from skimage import color 
from statistics import mode
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize
from pytesseract import pytesseract
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request
                        #render_template helps to redirect to the first page

unmaped_dict = {1: 'very_negative', 2: 'negative', 3: 'neutral', 4: 'positive', 5: 'very_positive'}

app = Flask(__name__)
path_to_tesseract = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/', methods = ['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    extract_ocr(image_path)
    
    return extract_ocr(image_path)


def text(t):
    cv = CountVectorizer(max_features = 5, min_df = 5, max_df = 0.8)
    t = cv.fit_transform([t])

    with open("Voting_Classifier.pkl", 'rb') as file:
        Pkl_random_forest = pickle.load(file)
    
    pred = Pkl_random_forest.predict(t)
    pred = pred.map(unmaped_dict)

    print(pred)
    return pred

def img_pred(imagefile):
    img = imread(imagefile)
    
    try:
        nx, ny, nrgb = img.shape
        
    except:
        img = color.gray2rgb(img)
        nx, ny, nrgb= img.shape
        
    x_train2 = img.reshape(nx, ny, nrgb)
    resized_img = resize(x_train2, (128*4, 64*4))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)


    with open('KNN_img_Classifier.pkl', 'rb') as file:
        Pkl_knn_img = pickle.load(file)
    pred = (Pkl_knn_img.predict([fd]))

    # with open('Decision_Tree_img_Classifier.pkl', 'rb') as file:
    #     Pkl_dt_img = pickle.load(file)
    # pred.append(Pkl_dt_img.predict([fd]))

    # with open('Logistic_Regression_img_Classifier.pkl', 'rb') as file:
    #     Pkl_lr_img = pickle.load(file)
    # pred.append(Pkl_lr_img.predict([fd]))

    #p = mode(pred)


    if (pred == 1):
        return 'Very Negative'
    elif (pred == 2):
        return 'Negative'
    elif (pred == 3):
        return 'Neutral'
    elif (pred == 4):
        return 'Positive'
    else:
        return 'Very Positive'
    

def extract_ocr(imagefile):
    # pytesseract.tesseract_cmd = path_to_tesseract   # Providing the tesseract executable location to pytesseract library
    # tex = pytesseract.image_to_string(imagefile)    # Passing image object to image_to_string() - will extract text from image

    # t = tex[:-1]

    # return text(t)
    return img_pred(imagefile)


if __name__ == '__main__':
    app.run(port = 3000, debug = True)
