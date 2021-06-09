FROM tensorflow/tensorflow:2.4.0

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy logic
COPY model ./model
COPY train_and_test ./train_and_test
COPY train.py .

# run
CMD ["python", "train.py"]

