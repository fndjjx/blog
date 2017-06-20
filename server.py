#coding:utf-8
import time
from flask import Flask, url_for, redirect, flash
from config import DevConfig
from flask.ext.sqlalchemy import SQLAlchemy
from sqlalchemy import func
from flask import Flask, render_template
from flask.ext.bootstrap import Bootstrap
from flask.ext.wtf import Form
from flask.ext.pagedown import PageDown
from wtforms import StringField, SubmitField, TextAreaField
from flask.ext.pagedown.fields import PageDownField
from markdown import markdown
import bleach


app = Flask(__name__)
app.config.from_object(DevConfig)
db = SQLAlchemy(app)
bootstrap = Bootstrap(app)
pagedown = PageDown(app)

#view
def sidebar_data():
    recent = Post.query.order_by(Post.publish_date.desc()).limit(5).all()
    return recent

@app.route("/")
def index():
    posts = Post.query.all()
    sidebar = sidebar_data()
    return render_template("index.html", posts=posts, sidebar=sidebar)

@app.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    print(post.html_text)
    sidebar = sidebar_data()
    
    return render_template("post.html", post=post, sidebar=sidebar)

@app.route("/edit", methods=['GET', 'POST'])
def edit():
    form = articleForm()
    if form.validate_on_submit():
        title = form.title.data
        content = form.content.data
        id = save_to_database(title, content) 
        flash('新文章已添加: {}'.format(title))
        return redirect(url_for('post', post_id=id))
    return render_template("edit.html", form=form)


#form
class articleForm(Form):
    title = StringField()
    #content = TextAreaField()
    content = PageDownField()
    submit = SubmitField('提交')


#model
class User(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(255))
    password = db.Column(db.String(255))
    posts = db.relationship(
        'Post',
        backref = 'user',
        lazy = 'dynamic'
    )

    def __init__(self, username):
        self.username = username

    def __repr__(self):
        return "user: {}".format(self.username)


class Post(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    title = db.Column(db.String(255))
    text = db.Column(db.Text())
    publish_date = db.Column(db.DateTime())
    user_id = db.Column(db.Integer(), db.ForeignKey('user.id'))
    html_text = db.Column(db.Text)

    def __init__(self, title):
        self.title = title

    def __repr__(self):
        return "post: {}".format(self.title)

    @staticmethod
    def on_changed_text(target, value, oldvalue, initiator):
        #allowed_tags=['a','ul','strong','p','h1','h2','h3', 'img', 'alt', 'src']
        #html_text = bleach.linkify(bleach.clean(
        #    markdown(value, output_format='html'), tags=allowed_tags, strip=True))
        html_text = markdown(value, output_format='html')
        target.html_text = html_text

def save_to_database(title, content):
    posts = Post.query.all()
    ids = [post.id for post in posts]
    if len(ids)>0:
        ids.sort()
        current_id = ids[-1] + 1
    else:
        current_id = 1

    post = Post(title)
    post.text = content
    post.publish_date = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    post.id = current_id

    db.session.add(post) 
    db.session.commit()
    return current_id

db.event.listen(Post.text, 'set', Post.on_changed_text)

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
