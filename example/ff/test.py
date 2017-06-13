from flask import Flask, render_template, request, url_for, redirect, Response, make_response, send_file, g
from flask.ext.bootstrap import Bootstrap
import pandas as pd
from datacleaner import autoclean

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        g.a = 1
        print(g.a)
        file = request.files['file']
        file.save("/tmp/"+file.filename)
        return redirect(url_for("upload_success", filename=file.filename))
    return render_template('test.html')

@app.route('/uploadsuccess/<filename>', methods=['GET', 'POST'])
def upload_success(filename):
    if request.method == 'POST':
        print("recevie form")
        #print(g.a)
        raw_data = pd.read_csv("/tmp/"+filename, error_bad_lines=False)
        clean_data = autoclean(raw_data)
        clean_data.to_csv("/tmp/"+filename, sep=',', index=False)
        
        return redirect(url_for('process_success', filename=filename))
    return render_template('upload.html')

@app.route('/processsuccess/<filename>')
def process_success(filename):
    print(filename)
    response = make_response(send_file("/tmp/"+filename))
    response.headers["Content-Disposition"] = "attachment; filename=myfiles.csv;"
    return response



if __name__ == "__main__":
    app.run(host='0.0.0.0')

