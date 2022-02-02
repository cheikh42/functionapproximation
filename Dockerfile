FROM jupyter/scipy-notebook 
RUN mkdir src
WORKDIR src/
COPY . .
RUN pip install -r requirements.txt



