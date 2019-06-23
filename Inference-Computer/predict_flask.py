from darkflow.net.build import TFNet
import cv2

from io import BytesIO
import time
import requests
from PIL import Image,ImageDraw
import numpy as np

# import sys
# from flask.ext.redis import Redis
from flask import Flask,Response,send_file,json
import random
import datetime

app = Flask(__name__)

options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1}

tfnet = TFNet(options)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)


counter = 0
color = ['black', 'gray', 'brown', 'red', 'orange', 'gold', 'yellow', 'green', 'teal', 'skyblue', 'blue', 'purple',
         'pink']
def handleBird():
    pass


def webapp():
    #while True:
        r = requests.get('http://192.168.15.237:5000/image.jpg') # replace with your ip address
        curr_img = Image.open(BytesIO(r.content))
        curr_img_cv2 = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)

        # uncomment below to try your own image
        #imgcv = cv2.imread('./sample/bird.png')
        result = tfnet.return_predict(curr_img_cv2)

        draw = ImageDraw.Draw(curr_img)
        for det in result:
            draw.rectangle([det['topleft']['x'], det['topleft']['y'],
                            det['bottomright']['x'], det['bottomright']['y']],
                           outline=random.choice(color))
            draw.text([det['topleft']['x'], det['topleft']['y'] - 13], det['label'] + ':' + str(det['confidence']),
                      fill=(255, 0, 0))
        #curr_img.save('obj_labeled/%i.jpg' % counter)
        #counter += 1
        #print('running again')

        byte_io = BytesIO()
        curr_img.save(byte_io, 'PNG')
        byte_io.seek(0)
        #time.sleep(1)
        #return send_file(byte_io, mimetype='image/png')
        return byte_io.read()

def webapp2():
    #while True:
        r = requests.get('http://192.168.15.237:5000/image.jpg') # replace with your ip address
        curr_img = Image.open(BytesIO(r.content))
        curr_img_cv2 = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)

        # uncomment below to try your own image
        #imgcv = cv2.imread('./sample/bird.png')
        result = tfnet.return_predict(curr_img_cv2)
        return result


def gen():
    """Video streaming generator function."""
    while True:
        frame = webapp()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/image')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/json')
def json_feed():
    return Response(json.dumps(webapp2(), cls=MyEncoder), content_type='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
