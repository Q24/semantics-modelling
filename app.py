#!flask/bin/python
from flask import Flask, jsonify
import numpy as np
import model_manager
from data import word_embedding_tools
app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]

@app.route('/<string:conv>', methods=['GET'])
def get_task(conv):
    print conv
    arr = np.array([.1,.2,.23], dtype='float32')
    return jsonify(arr.tolist())

if __name__ == '__main__':
    m = model_manager.ModelManager('test')


    app.run(debug=True)