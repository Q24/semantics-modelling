import requests
import numpy as np
'''
Handles the calls from a web-server using the Web Server Gateway Interface (WSGI) protocol
'''

def application(env, start_response):
    start_response('200 OK', [('Content-Type','text/html')])
    for key, value in env.iteritems():
        print key, value

    arr = np.array([0.0,1.2,-3.1], dtype='float32')
    print arr.tostring()
    print np.fromstring(arr.tostring(), dtype='float32')
    print 'FOUND: ', env['test']
    return [arr.tostring()]


if __name__ == '__main__':
    data = {"test1":"TRUE", "test2":"FALSE"}
    r = requests.get("http://0.0.0.0:8080",data)

    array = np.fromstring(r.content, dtype='float32')
    print array
    print