from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np

app = Flask(__name__)

dependencies = {
    'auc_roc': AUC
}

verbose_name = {
	       
 0: 'Affenpinscher',
 1: 'Afghan hound',
 2: 'Airedale terrier',
 3: 'Akita',
 4: 'Alaskan malamute',
 5: 'American eskimo dog',
 6: 'American foxhound',
 7: 'American staffordshire terrier',
 8: 'American water spaniel',
 9: 'Anatolian shepherd dog',
 10: 'Australian cattle dog',
 11: 'Australian shepherd',
 12: 'Australian terrier',
 13: 'Basenji',
 14: 'Basset hound',
 15: 'Beagle',
 16: 'Bearded collie',
 17: 'Beauceron',
 18: 'Bedlington terrier',
 19: 'Belgian malinois',
 20: 'Belgian sheepdog',
 21: 'Belgian tervuren',
 22: 'Bernese mountain dog',
 23: 'Bichon frise',
 24: 'Black and tan coonhound',
 25: 'Black russian terrier',
 26: 'Bloodhound',
 27: 'Bluetick coonhound',
 28: 'Border collie',
 29: 'Border terrier',
 30: 'Borzoi',
 31: 'Boston terrier',
 32: 'Bouvier des flandres',
 33: 'Boxer',
 34: 'Boykin spaniel',
 35: 'Briard',
 36: 'Brittany',
 37: 'Brussels griffon',
 38: 'Bull terrier',
 39: 'Bulldog',
 40: 'Bullmastiff',
 41: 'Cairn terrier',
 42: 'Canaan dog',
 43: 'Cane corso',
 44: 'Cardigan welsh corgi',
 45: 'Cavalier king charles spaniel',
 46: 'Chesapeake bay retriever',
 47: 'Chihuahua',
 48: 'Chinese crested',
 49: 'Chinese shar-pei',
 50: 'Chow chow',
 51: 'Clumber spaniel',
 52: 'Cocker spaniel',
 53: 'Collie',
 54: 'Curly-coated retriever',
 55: 'Dachshund',
 56: 'Dalmatian',
 57: 'Dandie dinmont terrier',
 58: 'Doberman pinscher',
 59: 'Dogue de bordeaux',
 60: 'English cocker spaniel',
 61: 'English setter',
 62: 'English springer spaniel',
 63: 'English toy spaniel',
 64: 'Entlebucher mountain dog',
 65: 'Field spaniel',
 66: 'Finnish spitz',
 67: 'Flat-coated retriever',
 68: 'French bulldog',
 69: 'German pinscher',
 70: 'German shepherd dog',
 71: 'German shorthaired pointer',
 72: 'German wirehaired pointer',
 73: 'Giant schnauzer',
 74: 'Glen of imaal terrier',
 75: 'Golden retriever',
 76: 'Gordon setter',
 77: 'Great dane',
 78: 'Great pyrenees',
 79: 'Greater swiss mountain dog',
 80: 'Greyhound',
 81: 'Havanese',
 82: 'Ibizan hound',
 83: 'Icelandic sheepdog',
 84: 'Irish red and white setter',
 85: 'Irish setter',
 86: 'Irish terrier',
 87: 'Irish water spaniel',
 88: 'Irish wolfhound',
 89: 'Italian greyhound',
 90: 'Japanese chin',
 91: 'Keeshond',
 92: 'Kerry blue terrier',
 93: 'Komondor',
 94: 'Kuvasz',
 95: 'Labrador retriever',
 96: 'Lakeland terrier',
 97: 'Leonberger',
 98: 'Lhasa apso',
 99: 'Lowchen',
 100: 'Maltese',
 101: 'Manchester terrier',
 102: 'Mastiff',
 103: 'Miniature schnauzer',
 104: 'Neapolitan mastiff',
 105: 'Newfoundland',
 106: 'Norfolk terrier',
 107: 'Norwegian buhund',
 108: 'Norwegian elkhound',
 109: 'Norwegian lundehund',
 110: 'Norwich terrier',
 111: 'Nova scotia duck tolling retriever',
 112: 'Old english sheepdog',
 113: 'Otterhound',
 114: 'Papillon',
 115: 'Parson russell terrier',
 116: 'Pekingese',
 117: 'Pembroke welsh corgi',
 118: 'Petit basset griffon vendeen',
 119: 'Pharaoh hound',
 120: 'Plott',
 121: 'Pointer',
 122: 'Pomeranian',
 123: 'Poodle',
 124: 'Portuguese water dog',
 125: 'Saint bernard',
 126: 'Silky terrier',
 127: 'Smooth fox terrier',
 128: 'Tibetan mastiff',
 129: 'Welsh springer spaniel',
 130: 'Wirehaired pointing griffon',
 131: 'Xoloitzcuintli',
 132: 'Yorkshire terrier'       
           }

# Select model
model = load_model('model.h5', compile=False)

model.make_predict_function()

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(299,299))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 299,299,3)

	predict_x=model.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return verbose_name[classes_x[0]]

# routes

@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    
@app.route("/index", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/tests/" + img.filename	
		img.save(img_path)

		predict_result = predict_label(img_path)

	return render_template("prediction.html", prediction = predict_result, img_path = img_path)

@app.route("/performance")
def performance():
	return render_template('performance.html')
    
@app.route("/chart")
def chart():
	return render_template('chart.html') 
if __name__ =='__main__':
	app.run(debug = True)
