
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from gevent import monkey
monkey.patch_all()
import time
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('my event', namespace='/test')
def handle_my_custom_event(message):
    print("connected")
    #emit('connect', {'num': 5, 'kind': 'apple', 'message': message['data']}, json=True)
    emit("connect","sb")
    time.sleep(3)
    for i in range(100):
        time.sleep(1)
        emit("connect",str(i))

if __name__ == '__main__':
    socketio.run(app,host='0.0.0.0')
