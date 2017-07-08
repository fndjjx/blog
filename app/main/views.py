#coding:utf-8
import sys
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
from . import main
from .forms import ArticleForm, LoginForm
from .. import db, bootstrap, pagedown, login_manager
from ..models import User, Post


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

@main.route("/")
@main.route("/index")
def index():
    print(current_user)
    posts = Post.query.all()
    sidebar = sidebar_data()
    sidebar_tag = sidebar_tag_data()
    return render_template("index.html", posts=posts, sidebar=sidebar, sidebar_tag=sidebar_tag)

@main.route("/tag/<tag>")
def tag(tag):
    posts = Post.query.filter(Post.tag.like("%"+tag+"%")).all()
    sidebar = sidebar_data()
    sidebar_tag = sidebar_tag_data()
    return render_template("tag.html", posts=posts, sidebar=sidebar, sidebar_tag=sidebar_tag)

@main.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    print(post.html_text)
    sidebar = sidebar_data()
    sidebar_tag = sidebar_tag_data()
    
    return render_template("post.html", post=post, sidebar=sidebar, sidebar_tag=sidebar_tag)

@main.route("/add", methods=['GET', 'POST'])
@login_required
def add_article():
    form = ArticleForm()
    if form.validate_on_submit():
        title = form.title.data
        content = form.content.data
        tag = form.tag.data
        id = save_to_database(title, content, tag) 
        flash('新文章已添加: {}'.format(title))
        return redirect(url_for('.post', post_id=id))
    return render_template("edit.html", form=form)

@main.route("/edit/<int:post_id>", methods=['GET', 'POST'])
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
        return redirect(url_for('.post', post_id=id))
    form.content.data = post.text
    form.title.data = post.title
    form.tag.data = post.tag
    return render_template("edit.html", form=form)

@main.route("/delete/<int:post_id>", methods=['GET', 'POST'])
@login_required
def delete_article(post_id):
    post = Post.query.get_or_404(post_id)
    db.session.delete(post)
    db.session.commit()
    flash('文章已删除')
    return json.dumps({"status": 302, "location": "/index"})


@main.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        user = User.query.filter_by(username=username).first()
        if user:
            login_user(user) 
            print(current_user)
            return redirect(request.args.get('next') or url_for('.index'))
        flash("Invalid user")
    return render_template('login.html', form=form)

@main.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You logout')
    return redirect(url_for('.index'))

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



