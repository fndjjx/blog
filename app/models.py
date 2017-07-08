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
from . import db, login_manager




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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

db.event.listen(Post.text, 'set', Post.on_changed_text)

