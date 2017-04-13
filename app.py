#! /usr/bin/env python

"""app.py: Launches web app (Flask-based) to accept input and displays final cluster counts."""

# imports
import sys
import io
import csv
from flask import Flask, make_response, request
import numpy as np
from kmeans import *

# create the application instance
app = Flask(__name__)


@app.route('/k-means')
def html_form():
    """Allow user to input csv file and number for k.

    Returns interactive options on the webpage for csv upload, k and submit.

    """
    return """
        <html>
            <body>
                <h1>Perform KMeans Algorithm</h1>
                <form action="/k-means-result" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    K: <input type="number" name="k"><br>
                    <input type="submit" />
                </form>
            </body>
        </html>
    """

@app.route('/k-means-result', methods=["POST"])
def process_input_kmeans():
    """Process user inputs through the kmeans algorithm and display output.

    Receives csv file and number for k through http requests.

    Displays a string representation of a sorted (ascending) list of the counts
    of how many observations are in each cluster.

    """
    f = request.files['data_file']
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    k = int(request.form['k'])
    observations = []
    for row in csv_input:
    	row = [int(i) for i in row]
    	observations.append(row)
    result = kmeans(np.array(observations), k)
    return make_response(result)


# Run the application using the port specified in command line by user
if __name__=='__main__':
	app.run(port=sys.argv[1])