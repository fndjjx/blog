from flask import Flask
from config import DevConfig
from flask.ext.sqlalchemy import SQLAlchemy
from sqlalchemy import func
from flask import Flask, render_template
from flask.ext.bootstrap import Bootstrap


app = Flask(__name__)
app.config.from_object(DevConfig)
db = SQLAlchemy(app)
bootstrap = Bootstrap(app)

#view
@app.route("/<int:page>")
def home(page=1):
    posts = Post.query.order_by(Post.publish_date.desc()).paginate(page, 40)
    recent, top_tags = sidebar_data()

    return render_template("index.html", posts=posts, recent=recent, top_tags=top_tags)

def sidebar_data():
    recent = Post.query.order_by(Post.publish_date.desc()).limit(5).all()
    top_tags = db.session.query(Tag, func.count(tags.c.post_id).label('total')).join(tags).group_by(Tag).order_by('total DESC').limit(5).all()
    return recent, top_tags

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template("post.html", post=post)

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

tags = db.Table('post_tags',
    db.Column('post_id', db.Integer, db.ForeignKey('post.id')),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'))
)

class Post(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    title = db.Column(db.String(255))
    text = db.Column(db.Text())
    publish_date = db.Column(db.DateTime())
    user_id = db.Column(db.Integer(), db.ForeignKey('user.id'))
    comments = db.relationship(
        'Comment',
        backref = 'post',
        lazy = 'dynamic'
    )
    tags = db.relationship(
        'Tag',
        secondary = tags,
        backref = db.backref('posts', lazy='dynamic')
    )

    def __init__(self, title):
        self.title = title

    def __repr__(self):
        return "post: {}".format(self.title)

class Comment(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(255))
    text = db.Column(db.Text())
    publish_date = db.Column(db.DateTime())
    post_id = db.Column(db.Integer(), db.ForeignKey("post.id"))


    def __repr__(self):
        return "comment: {}".format(self.text)



class Tag(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    title = db.Column(db.String(255))

    def __init__(self, title):
        self.title = title

    def __repr__(self):
        return "tag : {}".format(self.title)
if __name__=="__main__":
    #db.create_all()
    app.run(host="0.0.0.0", debug=True)
    #print(User.query.get(1))
    #user1 = User("leiyi")
    #db.session.add(user1)
    #db.session.commit()
    #print(User.query.get(1))
    #post1 = Post("a")
    #post1.user_id = user1.id
    #post1.text = "asdfsg"
    #db.session.add(post1)
    #db.session.commit()
    #print(user1.username)
    #post2 = Post("b")

    #user1.posts.append(post2)
    #db.session.add(user1)
    #db.session.commit()
    #uu = User.query.get(1)
    #print(list(uu.posts))
