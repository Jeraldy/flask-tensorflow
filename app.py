from flask import Flask,render_template,jsonify,request
from werkzeug import secure_filename
import base64,os,model,cv2

app = Flask(__name__)
UPLOAD_FOLDER='static'

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results",methods=['GET','POST'])
def results():
    if request.method == 'POST':
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
        image = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))

        return jsonify(model.predict(app.graph,image))

    return "Error"

app.graph = model.load_graph('static/retrained_graph.pb') 

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True,use_reloader=False)
