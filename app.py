from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pathlib
app = Flask(__name__)

def predict_label(img_path, model):
	dic_ = {0: 'Tumour', 1: 'No-Tumour'}
	i = image.load_img(img_path, target_size=(128,128))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 128,128,3)
	p = model.predict(i)
	if p[0][0] >= 0.5 :
		return dic_[1]
	else:
		return dic_[0]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Final Project Brain Tumour Detection...."


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	dir_ = pathlib.Path(r'C:\Users\hp\Desktop\Brain_Tumour_Final_Project\Final_Repo\model.h5')
	model = load_model(dir_)
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename
		img.save(img_path)
		p = predict_label(img_path, model)

	return render_template("index.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)

