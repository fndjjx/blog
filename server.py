#coding:utf-8
import time
from flask import Flask, url_for, redirect, flash, request
from config import DevConfig
from flask.ext.sqlalchemy import SQLAlchemy
from sqlalchemy import func
from flask import Flask, render_template
from flask.ext.bootstrap import Bootstrap
from flask.ext.wtf import Form
from flask.ext.pagedown import PageDown
from wtforms import StringField, SubmitField, TextAreaField, PasswordField, BooleanField
from flask.ext.pagedown.fields import PageDownField
from markdown import markdown
from markdown import Markdown
import bleach
import json

from flask.ext.login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user


app = Flask(__name__)
app.config.from_object(DevConfig)
db = SQLAlchemy(app)
bootstrap = Bootstrap(app)
pagedown = PageDown(app)
login_manager = LoginManager(app)
login_manager.session_protection = 'strong'
login_manager.login_view = 'login'

#view
def sidebar_data():
    recent = Post.query.order_by(Post.publish_date.desc()).limit(5).all()
    return recent

def sidebar_tag_data():
    all_tag = Post.query.with_entities(Post.tag).all()
    all_tag_pair_list = []
    for tag_pair in all_tag:
        all_tag_pair_list.extend(tag_pair)
    all_tag_pair_list = list(filter(lambda x:x, all_tag_pair_list))
    all_tag_list = []
    for tag_pair in all_tag_pair_list:
        all_tag_list.extend(tag_pair.split(" "))
    all_tag_list = list(set(all_tag_list))
    print(all_tag_list)
   
    return all_tag_list

@app.route("/")
@app.route("/index")
def index():
    print(current_user)
    posts = Post.query.all()
    sidebar = sidebar_data()
    sidebar_tag = sidebar_tag_data()
    return render_template("index.html", posts=posts, sidebar=sidebar, sidebar_tag=sidebar_tag)

@app.route("/tag/<tag>")
def tag(tag):
    posts = Post.query.filter(Post.tag.like("%"+tag+"%")).all()
    sidebar = sidebar_data()
    sidebar_tag = sidebar_tag_data()
    return render_template("tag.html", posts=posts, sidebar=sidebar, sidebar_tag=sidebar_tag)

@app.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    print(post.html_text)
    sidebar = sidebar_data()
    sidebar_tag = sidebar_tag_data()
    
    return render_template("post.html", post=post, sidebar=sidebar, sidebar_tag=sidebar_tag)

@app.route("/add", methods=['GET', 'POST'])
@login_required
def add_article():
    form = ArticleForm()
    if form.validate_on_submit():
        title = form.title.data
        content = form.content.data
        tag = form.tag.data
        id = save_to_database(title, content, tag) 
        flash('新文章已添加: {}'.format(title))
        return redirect(url_for('post', post_id=id))
    return render_template("edit.html", form=form)

@app.route("/edit/<int:post_id>", methods=['GET', 'POST'])
@login_required
def edit_article(post_id):
    post = Post.query.get_or_404(post_id)
    form = ArticleForm()
    if form.validate_on_submit():
        title = form.title.data
        content = form.content.data
        tag = form.tag.data
        id = save_to_database(title, content, tag, post_id)
        flash('文章已修改: {}'.format(title))
        return redirect(url_for('post', post_id=id))
    form.content.data = post.text
    form.title.data = post.title
    form.tag.data = post.tag
    return render_template("edit.html", form=form)

@app.route("/delete/<int:post_id>", methods=['GET', 'POST'])
@login_required
def delete_article(post_id):
    post = Post.query.get_or_404(post_id)
    db.session.delete(post)
    db.session.commit()
    flash('文章已删除')
    return json.dumps({"status": 302, "location": "/index"})


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        user = User.query.filter_by(username=username).first()
        if user:
            login_user(user) 
            print(current_user)
            return redirect(request.args.get('next') or url_for('index'))
        flash("Invalid user")
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You logout')
    return redirect(url_for('index'))



#form
class ArticleForm(Form):
    title = StringField()
    tag = StringField()
    #content = TextAreaField()
    content = PageDownField()
    submit = SubmitField('提交')


class LoginForm(Form):
    username = StringField("Name")
    password = PasswordField("Password")
    submit = SubmitField('提交')



#model
class User(db.Model, UserMixin):
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
    tag = db.Column(db.String(255))
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
        #html_text = markdown(value, output_format='html')
        myexts = ['extra', 'abbr', 'attr_list', 'def_list', 'fenced_code', 'footnotes', 'tables', 
                  'smart_strong', 'admonition', 'codehilite', 'headerid', 'meta', 'nl2br', 
                  'sane_lists', 'smarty', 'toc', 'mdx_math']

        md = Markdown(extensions=myexts)
        target.html_text = md.convert(value)

def save_to_database(title, content, tag, post_id=None):
    if post_id:
        current_id = post_id
        post = Post.query.filter_by(id=current_id).first()
    else:
        posts = Post.query.all()
        ids = [post.id for post in posts]
        if len(ids)>0:
            ids.sort()
            current_id = ids[-1] + 1
        else:
            current_id = 1

        post = Post(title)
    post.title = title
    post.text = content
    post.publish_date = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    post.id = current_id
    post.tag = tag

    db.session.add(post) 
    db.session.commit()
    return current_id

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

db.event.listen(Post.text, 'set', Post.on_changed_text)

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
