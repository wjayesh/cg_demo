FROM tensorflow/tensorflow:2.4.0

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy logic
COPY fetch_and_preprocess.py .

# run
CMD ["python", "fetch_and_preprocess.py"]