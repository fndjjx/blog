from flask import Flask, render_template
from flask import request, session
from flask import redirect, url_for, flash
from flask import current_app, g
from flask.ext.bootstrap import Bootstrap
from flask.ext.wtf import Form
from wtforms import StringField, SubmitField
from wtforms.validators import Required


app = Flask(__name__)
app.config["SECRET_KEY"] = "haha"
bootstrap = Bootstrap(app)

@app.route("/index/<name>")
def index(name):
    return render_template("index.html",name=name)

@app.route("/user/<para>")
def hello(para):
    return "<h1>hello {}</h1> <p>your browser is {} {}</p>".format(para, request.headers.get("User-Agent"),1)

@app.errorhandler(404)
def error(e):
    return "<h1>not found haha</h1>"


@app.route("/",methods=["GET","POST"])
def index2():
    name = None
    form = NameForm()
    if form.validate_on_submit():
         session['name'] = form.name.data
         form.name.data = ""
         flash("nice {}".format(session.get("name")))
         return redirect(url_for("index2"))
    return render_template("index2.html", form=form, name=session.get("name"))

class NameForm(Form):
    name = StringField("name?", validators=[Required()])
    submit = SubmitField('s')

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
